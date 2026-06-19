with open("src/components/KineticMap/KineticMap.tsx", "r") as f:
    content = f.read()

start_str = "const B2S: [number,number,number,number][] = ["
end_str = "];\n\nconst bx2sx = (bx: number): number => {"

start_idx = content.find(start_str)
end_idx = content.find(end_str)

if start_idx != -1 and end_idx != -1:
    new_b2s = """const B2S: [number,number,number,number][] = [
  [100,   250,  20,  50 ],    // ORIGIN → CSMT sw_in
  [250,   430,  50,  130],    // CSMT station zone
  [430,   880,  130, 330],    // CSMT → DADAR segment
  [880,   1060, 330, 420],    // DADAR station zone
  [1060,  2560, 420, 870],    // DADAR → KALYAN segment
  [2560,  2740, 870, 970],    // KALYAN station zone
  [2740,  3190, 970, 1150],   // KALYAN → AMBERNATH segment (inc switch)
  [3190,  3370, 1150, 1220],  // AMBERNATH station zone
  [3370,  4120, 1220, 1370],  // AMBERNATH → TITWALA segment
  [4120,  4300, 1370, 1440],  // TITWALA station zone
  [4300,  5050, 1440, 1590],  // TITWALA → ATGAON segment
  [5050,  5230, 1590, 1660],  // ATGAON station zone
  [5230,  6130, 1660, 1860],  // ATGAON → KASARA segment
  [6130,  6310, 1860, 1940],  // KASARA station zone
  [6310,  7660, 1940, 2388],  // KASARA switch + ghat + IGATPURI switch
  [7660,  7840, 2388, 2468],  // IGATPURI station zone
  [7840,  9340, 2468, 2868],  // IGATPURI → DEVLALI segment
  [9340,  9520, 2868, 2938],  // DEVLALI station zone
  [9520,  10120, 2938, 3088], // DEVLALI → NASHIK segment
  [10120, 10300, 3088, 3178], // NASHIK station zone
  [10300, 11150, 3178, 3378], // NASHIK → NANDGAON segment
  [11150, 11310, 3378, 3448], // NANDGAON crossing loop zone
  [11310, 11930, 3448, 3668], // NANDGAON → LASALGAON segment
  [11930, 12090, 3668, 3738], // LASALGAON crossing loop zone
  [12090, 12760, 3738, 3968], // LASALGAON → MANMAD segment
  [12760, 13090, 3968, 4048], // MANMAD station zone
  [13090, 13300, 4048, 4140], // MANMAD → DEST
"""
    
    content = content[:start_idx] + new_b2s + content[end_idx:]
    with open("src/components/KineticMap/KineticMap.tsx", "w") as f:
        f.write(content)
    print("Fixed B2S successfully.")
else:
    print("Could not find boundaries!")

