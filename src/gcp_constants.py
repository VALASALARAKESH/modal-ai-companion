import modal
gcp_hmac_secret = modal.Secret.from_name(
"googlecloud-secret",
required_keys=["GOOGLE_ACCESS_KEY_ID", "GOOGLE_ACCESS_KEY_SECRET"]
)
GCP_PUBLIC_IMAGE_BUCKET = 'ai-companion-bucket-image'
GCP_CHAT_BUCKET = 'ai-companion-bucket-chat'

#GCP_PUBLIC_IMAGE_BUCKET = 'coqui-samples'
#GCP_CHAT_BUCKET = 'modal-agent-chat-test'