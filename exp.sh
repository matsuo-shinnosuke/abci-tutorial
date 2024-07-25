for lr in 0.1 0.01 0.01
do
    python src/main.py --lr=$lr --output_dir='result/lr='${lr}'/'
done
