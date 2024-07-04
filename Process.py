import json
import csv

def get_philadelphia_business_details(business_file):
    philadelphia_business_details = {}
    with open(business_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                business = json.loads(line)
                if business['city'] == "Philadelphia":
                    if not business['categories']: 
                        category = 'unclear'
                    else: 
                        category = str(business['categories'])
                    philadelphia_business_details[business['business_id']] = {
                        'name': business['name'],
                        'categories': category
                    }
            except json.JSONDecodeError:
                continue
    return philadelphia_business_details

def filter_reviews_and_write_csv(review_file, business_details, output_csv_file):
    with open(review_file, 'r', encoding='utf-8') as file:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            # fieldnames = ['business_id', 'business_name', 'categories', 'review_id', 'review', 'stars', 'useful', 'funny', 'cool', 'summary']
            fieldnames = ['summary']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for line in file:
                try:
                    review = json.loads(line)
                    if review['business_id'] in business_details:
                        summary = ".".join([f"This a review for the business {business_details[review['business_id']]['name']}, and it has been categorized as {business_details[review['business_id']]['categories']}"
                                            , f"The review: {review['text']} and the customer gave the business {business_details[review['business_id']]['name']} a rating of {review['stars']} stars"
                                            , f"{review['useful']} people found the review useful, {review['funny']} people found the review funny, and {review['cool']} people found the review cool."
                                            ])

                        # writer.writerow({
                        #     'business_id': review['business_id'],
                        #     'business_name': business_details[review['business_id']]['name'],
                        #     'categories': business_details[review['business_id']]['categories'],
                        #     'review_id': review['review_id'],
                        #     'review': review['text'],
                        #     'stars': review['stars'],
                        #     'useful': review['useful'],
                        #     'funny': review['funny'],
                        #     'cool': review['cool'],
                        #     'summary': summary
                        # })
                        
                        writer.writerow({
                            'summary': summary
                        })
                except json.JSONDecodeError:
                    continue

def filter_tips_and_write_csv(tip_file, business_details, output_csv_file):
    with open(tip_file, 'r', encoding='utf-8') as file:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            # fieldnames = ['business_id', 'business_name', 'categories', 'tips', 'compliment', 'summary']
            fieldnames = ['summary']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for line in file:
                try:
                    tip = json.loads(line)
                    if tip['business_id'] in business_details:

                        summary = ".".join([f"This a tip for the business {business_details[tip['business_id']]['name']}, and it has been categorized as {business_details[tip['business_id']]['categories']}"
                                            ,f"The tip says '{tip['text']}'' and {tip['compliment_count']} customer agreed with this tip."])
                                            
                        writer.writerow({
                            'summary': summary
                        })
                        
                        # writer.writerow({
                        #         'business_id': tip['business_id'],
                        #         'business_name': business_details[tip['business_id']]['name'],
                        #         'categories': business_details[tip['business_id']]['categories'],
                        #         'tips': tip['text'],
                        #         'compliment': tip['compliment_count'],
                        #         'summary': summary
                        #     })
                except json.JSONDecodeError:
                    continue

# File paths
business_file = "C:/Users/zhuan/Downloads/yelp_dataset/yelp_academic_dataset_business.json"
review_file = "C:/Users/zhuan/Downloads/yelp_dataset/yelp_academic_dataset_review.json"
tip_file = "C:/Users/zhuan/Downloads/yelp_dataset/yelp_academic_dataset_tip.json"
output_file = "C:/Users/zhuan/Desktop/Cornell Tech/School Work/Yelp RAG/filtered_reviews_dense.csv"
output_file_tip = "C:/Users/zhuan/Desktop/Cornell Tech/School Work/Yelp RAG/filtered_tips_dense.csv"

# Processing
business_details = get_philadelphia_business_details(business_file)
filter_reviews_and_write_csv(review_file, business_details, output_file)
filter_tips_and_write_csv(tip_file, business_details, output_file_tip)