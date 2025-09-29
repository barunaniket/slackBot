use secrecy::{ExposeSecret, Secret};
// We need to bring SlackEventCallbackBody into scope to use it
use slack_morphism::{SlackEventCallbackBody, prelude::*};
use std::env;
use std::sync::Arc;

// Configuration loading remains the same
fn config() -> Result<AppConfig, Box<dyn std::error::Error + Send + Sync>> {
    dotenv::dotenv().ok();
    let slack_bot_token: Secret<String> = Secret::new(env::var("SLACK_BOT_TOKEN")?);
    let slack_app_token: Secret<String> = Secret::new(env::var("SLACK_APP_TOKEN")?);
    Ok(AppConfig {
        slack_bot_token,
        slack_app_token,
    })
}

pub struct AppConfig {
    pub slack_bot_token: Secret<String>,
    pub slack_app_token: Secret<String>,
}

// The event handler function. Its signature is now fully specified.
async fn on_socket_mode_event(
    event: SlackSocketModeEvent<SlackEventCallbackBody>, // The correct, full type for the event
    client: Arc<SlackClient<SlackClientHyperConnector>>,
    _states: SlackClientEventsUserState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Received event: {:?}", event.event);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = config()?;
    println!("-> Initializing Slack client...");

    let connector = SlackClientHyperConnector::new();
    let client = Arc::new(SlackClient::new(connector));

    // The correct method is `.with_push_events()`
    let on_event_callbacks = SlackSocketModeListenerCallbacks::new()
        .with_push_events(on_socket_mode_event);

    let listener_environment = Arc::new(
        SlackClientEventsListenerEnvironment::new(client.clone())
    );

    let listener = SlackClientSocketModeListener::new(
        &SlackClientSocketModeConfig::new(),
        listener_environment,
        on_event_callbacks,
    );

    // We must create a `SlackApiToken` from our secret string.
    let app_token_value: SlackApiTokenValue = config.slack_app_token.expose_secret().clone().into();
    let app_token = SlackApiToken::new(app_token_value);

    println!("-> Connecting to Slack...");

    // The listener takes a reference to the `SlackApiToken`.
    listener.listen_for(&app_token).await?;

    Ok(())
}