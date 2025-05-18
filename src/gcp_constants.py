import modal
gcp_hmac_secret = modal.Secret.from_name(
"googlecloud-secret",
required_keys=["GOOGLE_ACCESS_KEY_ID", "GOOGLE_ACCESS_KEY_SECRET"]
)
GCP_PUBLIC_IMAGE_BUCKET = 'ai-companion-bucket-image'
GCP_CHAT_BUCKET = 'ai-companion-bucket-chat'




GCP_BUCKET_ENDPOINT_URL = "https://storage.googleapis.com"