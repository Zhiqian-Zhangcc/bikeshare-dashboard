import random
STEP = 5        
PEOPLE = 4      
dishes = [
      "火锅", "麻辣烫", "黄焖鸡", "兰州拉面", "螺蛳粉",
      "沙县小吃", "肯德基", "烤肉", "寿司", "披萨",
      "汉堡", "炒饭", "米线", "饺子", "凉皮",
      "麻辣香锅", "烧烤", "冒菜", "酸菜鱼", "奶茶",
  ]
weight = {d: 5.0 for d in dishes}   
mentioned = []
for i in range(PEOPLE):
      while True:
          d = input(f"室友 {i + 1} 想吃啥: ").strip()
          if d in weight:
              mentioned.append(d)
              break
          print(f"菜单里没有「{d}」,重新输入")

  # ===== 2. 被提到的菜,权重提高 =====
for d in mentioned:
      weight[d] += STEP

mentioned_unique = set(mentioned)                       
mentioned_total = sum(weight[d] for d in mentioned_unique)  # 它们现在的权重和
remaining = 100 - mentioned_total                        # 剩下的权重
# ===== 3. 剩下的权重平均分给"没人提到"的菜 =====
unmentioned = [d for d in dishes if d not in mentioned_unique]
if remaining < 0:
      print("想吃得太集中,权重超过 100% 了,请把 STEP 调小")
elif unmentioned:
      each = remaining / len(unmentioned)
      for d in unmentioned:
          weight[d] = each

  # ===== 4. 展示结果 shao=====
print("\n===== 最终权重 =====")
for d in dishes:
      mark = "  ← 有人想吃" if d in mentioned_unique else ""
      print(f"{d:<6} {weight[d]:5.2f}%{mark}")

  # ===== 5. 按权重随机决定今天吃什么 =====
chosen = random.choices(dishes, weights=[weight[d] for d in dishes], k=1)[0]
print(f"\n>>> 今天就吃: {chosen} !")