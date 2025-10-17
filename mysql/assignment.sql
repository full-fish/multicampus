use madang;
INSERT INTO book (bookid, bookname, publisher, price) VALUES
(1, '축구의 역사', '굿스포츠', 7000),
(2, '축구아는 여자', '나무수', 13000),
(3, '축구의 이해', '대한미디어', 22000),
(4, '골프 바이블', '대한미디어', 35000),
(5, '피겨 교본', '굿스포츠', 8000),
(6, '역도 단계별기술', '굿스포츠', 6000),
(7, '야구의 추억', '이상미디어', 20000),
(8, '야구를 부탁해', '이상미디어', 13000),
(9, '올림픽 이야기', '삼성당', 7500),
(10, 'Olympic Champions', 'Pearson', 13000);

INSERT INTO customer (custid, name, address, phone) VALUES
(1, '박지성', '영국 맨체스터', '000-5000-0001'),
(2, '김연아', '대한민국 서울', '000-6000-0001'),
(3, '장미란', '대한민국 강원도', '000-7000-0001'),
(4, '추신수', '미국 클리블랜드', '000-8000-0001'),
(5, '박세리', '대한민국 대전', NULL);

INSERT INTO orders (orderid, custid, bookid, saleprice, orderdate) VALUES
(1, 1, 1, 6000, '2014-07-01'),
(2, 2, 3, 21000, '2014-07-03'),
(3, 3, 5, 8000, '2014-07-03'),
(4, 4, 6, 6000, '2014-07-04'),
(5, 1, 3, 21000, '2014-07-05'),
(6, 2, 7, 20000, '2014-07-07'),
(7, 3, 2, 13000, '2014-07-07'),
(8, 4, 8, 13000, '2014-07-08'),
(9, 5, 10, 7000, '2014-07-09'),
(10, 3, 8, 13000, '2014-07-10');

-- ①가격이 20,000원 미만인 도서를 검색
select * from book where price<20000;
-- ②가격이 10,000원 이상 20,000 이하인 도서를 검색
select * from book where 10000 <= price and price <= 20000;
-- ③출판사가 ‘굿스포츠’ 혹은 ‘대한미디어’인 도서를 검색
select * from book where publisher = '굿스포츠' or publisher = '대한미디어';
-- ④출판사가 ‘굿스포츠’ 혹은 ‘대한미디어’가 아닌 도서를 검색
select * from book where publisher != '굿스포츠' and publisher != '대한미디어';
-- ⑤‘축구의 역사’를 출간한 출판사를 검색
select * from book where bookName = '축구의 역사';
-- ⑥도서이름에 ‘축구’가 포함된 출판사를 검색
select * from book where bookName like '%축구%';
-- ⑦도서이름이 여섯 글자인 도서를 검색
select * from book where char_length(bookName) = 6;
-- ⑧도서이름의 왼쪽 두 번째 위치에 ‘구’라는 문자열을 갖는 도서를 검색
select * from book where bookName like '_구%';
-- ⑨축구에 관한 도서 중 가격이 20,000원 이상인 도서를 검색
select * from book where bookName like '%축구%' and price >= 20000;
-- ⑩야구에 관한 책을 모두 구입하려면 필요한 금액 계산
select sum(price) from book where bookName like '%야구%';
-- ⑪ 도서를 가격 순으로 검색하고, 가격이 같으면 이름순으로 검색
select * from book order by price, bookName;
-- ⑫ 도서를 가격의 내림차순으로 검색하고 만약 가격이 같다면 출판사의 오름차순으로 검색
select * from book order by price desc, publisher asc;
-- ⑬ 주소가 우리나라나 영국인 선수정보 조회
select * from customer where address like '대한민국%' or address like '영국%';
-- ⑭고객이 주문한 도서의 총 판매액 조회
select sum(salePrice) from orders;
-- ⑮번 김연아 고객이 주문한 도서의 총 판매액 조회
select sum(salePrice) as '총 판매액' 
from customer as C join orders as O on C.custId = O.custId where C.custId = 2;
-- 16) 고객이 주문한 도서의 총 판매액, 평균값, 최저가, 최고가 조회
select sum(salePrice) as '총 판매액', avg(salePrice) as '평균값', 
		min(salePrice) as '최저가', max(salePrice) as '최고가' 
from orders;
-- 17) 마당서점의 도서 판매 건수 조회
select count(*) from orders;
-- 18) 고객별로 주문한 도서의 총 수량과 총 판매액 조회
select C.name, count(*) as '총 구매 도서 수량', sum(salePrice) as '총 판매액' 
from customer as C join orders as O on C.custId = O.custId group by C.name;
-- 19) 가격이 8,000원 이상인 도서를 구매한 고객에 대하여 고객별 주문 도서의 총 수량을 구하시오. 단, 두 권 이상 구매한 고객만 조회
select C.name as '고객명' , count(*) as '총 구매 도서 수량' 
from customer as C
join orders as O
on C.custId = O.custId
join book as B
on O.bookId = B.bookId
where B.price >= 8000
group by C.name
having count(*) >=2;
-- 20) 날짜별 총구매건수와 총판매액을 조회
select orderDate as '날짜', count(*) as '총구매건수', sum(salePrice) as '총판매액' 
from orders group by orderDate;
-- 21)총판매액이 20000이 넘는 날짜의 총 구매건수를 조회
select orderDate as '날짜', count(*) as '총구매건수', sum(salePrice) as '총판매액' 
from orders group by orderDate having sum(salePrice)>=20000;
-- 22) 가장 구매건수가 많은 날짜를 조회 구매건수가 같은 경우 가장 최근 날짜를 조회
select orderDate as '날짜', count(*) as '총구매건수' 
from orders group by orderDate order by count(*) desc, orderDate asc;