curl http://llms-backend.axgln.net/llms/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "你好，请简要介绍一下GPT-4o。"}
    ],
    "temperature": 0.7,
    "stream": true
  }'
