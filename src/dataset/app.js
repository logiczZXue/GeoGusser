import express from 'express';

const app = express();
const PORT = 1;

// 使用 Express 的 static 中间件提供 public 目录下的静态文件
app.use(express.static('public'));

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
