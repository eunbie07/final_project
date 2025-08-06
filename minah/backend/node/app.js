// Node.js에 기본으로 내장된 http 모듈을 불러옵니다.
const http = require('http');

// 애플리케이션이 실행될 포트를 3000번으로 설정합니다.
// Nginx 설정 파일의 proxy_pass 포트와 일치해야 합니다.
const port = 3002;

// HTTP 서버를 생성합니다.
// 이 서버는 요청이 들어올 때마다 실행될 함수를 가집니다.
const server = http.createServer((req, res) => {

  // HTTP 상태 코드를 200 (성공)으로 설정합니다.
  res.statusCode = 200;
  
  // 응답 헤더를 설정합니다. 한글이 깨지지 않도록 charset=utf-8을 추가합니다.
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  
  // 클라이언트(브라우저)에게 보낼 응답 메시지입니다.
  res.end('Node.js 서버가 성공적으로 응답합니다!');
});

// 위에서 설정한 port(3000번)에서 요청을 기다리도록 서버를 시작합니다.
server.listen(port, () => {
  // 서버가 성공적으로 시작되면 콘솔에 이 메시지를 출력합니다.
  console.log(`서버가 http://localhost:${port} 에서 실행 중입니다.`);
});