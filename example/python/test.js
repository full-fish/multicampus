// let pt = [37.5665, 126.9780]
// console.log(pt[0], pt[1])

// let d = { [pt]: "Seoul" }

// console.log(d)
// console.log(d[pt])


// function a(...i) {
//   console.log(i);
// }
// a([1, 2, 3])

function f() {
  s = 'lon'
  console.log(s);
}
let s = 'paris'
f()
console.log(s);
// 기본적으로 js는 전역 변수취급 파이썬은 global 써 줘야함
// js에서 내부 함수에 let해주면 지역 변수 취급 파이썬은 그냥 쓰면 지역 변수 취금

arr.filter((x) => x % 2 === 0).map((ele) => ele + 1)