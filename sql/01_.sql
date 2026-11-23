SELECT *
FROM ai_analyzer.classified_reviews
LIMIT 10;

SELECT 
	sentiment,
	count(*) AS group_count,
	round(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),2) AS percentage
FROM ai_analyzer.classified_reviews
GROUP BY sentiment;

SELECT 
	category ,
	count(*) AS group_count,
	round(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),2) AS percentage
FROM ai_analyzer.classified_reviews
GROUP BY category
ORDER BY percentage desc;

SELECT 
	sentiment,
	category,
	count(*) AS group_count,
	round(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY category),2) AS percentage
FROM ai_analyzer.classified_reviews
GROUP BY category , sentiment
ORDER BY category, sentiment;

SELECT 
	sentiment,
	review_score ,
	count(*) AS group_count,
	round(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY review_score),2) AS percentage
FROM ai_analyzer.classified_reviews
GROUP BY review_score , sentiment
ORDER BY review_score desc, sentiment;

SELECT problem,
	count(*) AS cnt_problem
FROM ai_analyzer.classified_reviews
WHERE sentiment = 'Negative' AND problem IS NOT NULL 
GROUP BY problem
ORDER BY cnt_problem desc
LIMIT 15;
