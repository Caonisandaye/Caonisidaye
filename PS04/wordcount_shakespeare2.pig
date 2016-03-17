-- wordcount_shakespeare2.pig
--
--
-- Find the top-20 words in Shakespeare, ignoring case
--
-- Clear the output directory location
--
rmf sorted_words2

--
-- Run the script

shakespeare = LOAD 's3://gu-anly502/ps04/Shakespeare.txt' as (line:chararray);

-- YOUR CODE GOES HERE
-- PUT YOUR RESULTS IN sorted_words20

STORE sorted_words20 INTO 'sorted_words2' USING PigStorage();
 
-- Get the results
--
fs -getmerge sorted_words2 wordcount_shakespeare.txt
