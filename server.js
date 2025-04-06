const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

app.use(express.json());
app.use(cors());
app.use(express.static('public'));

app.post('/chat', async (req, res) => {
    try {
        const userMessage = req.body.message;

        // 🔹 Forward user message to Python backend
        const response = await axios.post('http://127.0.0.1:5000/chat', { message: userMessage });

        // 🔹 Send Python's response back to frontend
        res.json({ reply: response.data.reply });
    } catch (error) {
        console.error("Error communicating with Python backend:", error);
        res.status(500).json({ reply: "Sorry, there was an error processing your request." });
    }
});

app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});
