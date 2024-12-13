import modal
gcp_hmac_secret = modal.Secret.from_name(
"gcp-secret-prod",
required_keys=["GOOGLE_ACCESS_KEY_ID", "GOOGLE_ACCESS_KEY_SECRET"]
)
GCP_PUBLIC_IMAGE_BUCKET = 'truluv-public'
GCP_CHAT_BUCKET = 'truluv-chats'

#GCP_PUBLIC_IMAGE_BUCKET = 'coqui-samples'
#GCP_CHAT_BUCKET = 'modal-agent-chat-test'

GCP_BUCKET_ENDPOINT_URL = "https://storage.googleapis.com"