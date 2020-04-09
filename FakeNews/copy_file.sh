#!/bin/bash

mkdir /media/samantha/1CBCF987BCF95C28/Samantha/FakeNewsData/NELA-GT/2018-03-04

mv ~/Documents/samDocs/Projects/FakeNews/datasets/SentimentScore/NELA-GT/2018-03-04/*_avg.csv /media/samantha/1CBCF987BCF95C28/Samantha/FakeNewsData/NELA-GT/2018-03-04

rm -rf ~/Documents/samDocs/Projects/FakeNews/datasets/SentimentScore/NELA-GT/2018-03-04

rm -rf ~/Documents/samDocs/Projects/FakeNews/datasets/articles/2018-03-04

mkdir ~/Documents/samDocs/Projects/FakeNews/datasets/SentimentScore/NELA-GT/2018-03-04

cp -r ~/Downloads/articles/2018-03-04 ~/Documents/samDocs/Projects/FakeNews/datasets/articles

