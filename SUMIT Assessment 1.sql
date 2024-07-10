use satyug;

select * from shopping;

select sum(purch_amt) as Total_Purch_amt from shopping;

select avg(purch_amt) as Avg_Purch_amt from shopping;

select count(distinct salesman_id) from shopping;

select count(customer_id) from shopping;

select max(purch_amt) as Max_Purch_amt from shopping;

select min(purch_amt) as Min_Purch_Amt from shopping;

select count(customer_id) from shopping
where grade >=1;