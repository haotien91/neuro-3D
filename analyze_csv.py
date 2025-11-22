import pandas as pd

# Set pandas display options to show full filenames
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

try:
    df = pd.read_csv('sub03_comparison.csv')
    
    print("=== Top 5 Minimum Chamfer Distance (Most Similar) ===")
    print(df.nsmallest(5, 'chamfer_distance')[['filename', 'chamfer_distance']].to_string(index=False))
    
    print("\n=== Top 5 Maximum Chamfer Distance (Most Different) ===")
    print(df.nlargest(5, 'chamfer_distance')[['filename', 'chamfer_distance']].to_string(index=False))

except FileNotFoundError:
    print("Error: sub03_comparison.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
