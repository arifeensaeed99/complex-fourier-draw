apt-get update && apt-get install -y libicu-dev

mkdir -p ~/.streamlit/

echo "[theme]
primaryColor = ‘#84a3a7’
backgroundColor = ‘#EFEDE8’
secondaryBackgroundColor = ‘#fafafa’
textColor= ‘#424242’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false

" > ~/.streamlit/config.toml
