import os
import requests
"""
curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer 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" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/orgs/ORG/personal-access-token-requests
"""

GITHUB_TOKEN="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"
def get_github_id():

	headers = {
	    'Accept': 'application/vnd.github+json',
	    'Authorization': 'Bearer {GITHUB_TOKEN}',
	    'X-GitHub-Api-Version': '2022-11-28',
	}

	return requests.get('https://api.github.com/orgs/ORG/personal-access-token-requests', headers=headers)

DATABERRY_API="d61703a8-6543-4500-8434-6b05467ec396"
ELEVENLABS_API_KEY="ed866abd23af31f1c74b249ce12fd198"
#print(get_github_id())
# Hugging FACE
HUGGINGFACE_TOKEN= "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"
HUGGINGFACE_EMAIL = "goldenkooy@gmail.com"
HUGGINGFACE_PASS = "Electroman75!."
OPENAI_API_KEY="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" # new code
BEARLY_API_KEY="bearly-sk-Ab6PwdBCbEt23UOU3StU6lh6Jk"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="ls__0dbbf9b0dea44c7fae6de2b778169486"
LANGCHAIN_PROJECT="K00B404"
# Required for agent example
SERPAPI_API_KEY="122c4483f3d4ea48293de5f5ffe5d52584090acdfd5d0e978cc5e471100bba64"
# 9c2d4a7a5311589c7764688f5c2a0b3b10cb39fb
   # Variables for Weaviate db.
vector_db_url= "https://some-endpoint.weaviate.network"
vector_db_api_key= "PykLgRQtcj4SD5NZ5S058gSsxDMEAIViR0hc"
# Helicone
helicone_api_base  = "https://anthropic.hconeai.com/v1"
helicone_api_key= "sk-jrjozvq-dofemii-rtdk6zy-hes5pla"
# Required for retrieval examples
SUPABASE_PRIVATE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJoaHBlemxlenVjam5vZXprbGdzIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTUwMDc2NTQsImV4cCI6MjAxMDU4MzY1NH0.CWhKtE5gCo5czpWT166U27jyqzRYKI16lS0sRWtnLeI"
SUPABASE_URL="https://rhhpezlezucjnoezklgs.supabase.co"

REPLICATE_API = "r8_f5Pub5TEERECcAjGeIE4HiY1tNiMfjJ36lCC5"
#PINECONE Vector storage db
PINECONE_ENV = "us-west1-gcp-free"
PINECONE_KEY = "97a95a93-3d0f-490e-8288-732063523f68"
# BING chat ( rember to copy yours cookies into : /content/Free-AUTO-GPT-with-NO-API/cookiesBing.json )
BINGCHAT_COOKIEPATH = "cookiesBing.json"

# Google Bard
BARDCHAT_TOKEN = "aQis6F_8a1oM3tjLEmwqvqRgr5ivLJfkrVwMwnmwnrHj_8j2ElhmGah8ts3c_1RCY8m65A."
# googlesearch
GOOGLE_SEARCH_API = "AIzaSyAHN9bea-FDmbg8jCqKAHbDJFrbdZP99Sg"
GOOGLE_SEARCH_ID = "api-project-975959468141"
REPLICATE_API_KEY ="r8_TwdWM5Jh3KdXHn2q1aAr5cew1XFKrLa10i1UI"

access_token = (
    "81928627_c009bf122ccf36ec3ba3e0ef748b07042c5e4217260042004a5934540cb61527"
)

# Init toolkit
#clickup_api_wrapper = ClickupAPIWrapper(access_token=access_token)
#toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup_api_wrapper)

{
  "token_type": "Bearer",
  "expires_in": 86400,
  "access_token": "rfrrhlWYPf2Xwd2l2oE-gU1_iWTDZODXrGvwfXTEM2UuP3apmojFloRnmFlNVuFIeUlAN5WZ",
  "scope": "photo offline_access",
  "refresh_token": "lw3Vy9nlS466lHammWHjDLlD"
}
#clickup
client_id ="-PV7mFpYLBOxgh9XC-y6SlK4"
client_secret =	"cVMiVlXHBMFkawdahLVSeXTDiz9HEX1-XptgdwMFK9dfU_W1"
login 	='tense-peafowl@example.com'
password ='Healthy-Ibex-89'

clickup = "Found team_id: {clickup_api_wrapper.team_id}.\nMost request require the team id, so we store it for you in the toolkit, we assume the first team in your list is the one you want. \nNote: If you know this is the wrong ID, you can pass it at initialization."

# Optional: For Tracing with LangSmith
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=YOUR_API_KEY
# LANGCHAIN_PROJECT=nextjs-starter

# Optional: Other model keys
# ANTHROPIC_API_KEY="YOUR_API_KEY"

if __name__ == "__main__":
    print("github"+get_github_id())
    print(HUGGINGFACE_TOKEN)
    print(OPENAI_API_KEY)
    print(BINGCHAT_COOKIEPATH)
    print(BARDCHAT_TOKEN)
