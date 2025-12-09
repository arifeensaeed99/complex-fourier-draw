export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=true

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
