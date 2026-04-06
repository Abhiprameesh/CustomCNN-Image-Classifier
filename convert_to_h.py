import os

def tflite_to_c_array(tflite_path, header_path):
    # Read the TFLite file as binary
    with open(tflite_path, 'rb') as f:
        file_data = f.read()
        
    array_name = "model_quant_tflite"
        
    # Format the data as hex values
    hex_array = [format(byte, '#04x') for byte in file_data]
    
    # Write the C Header file
    with open(header_path, 'w') as f:
        f.write("#ifndef MODEL_H\n")
        f.write("#define MODEL_H\n\n")
        f.write(f"// Generated from {tflite_path}\n")
        f.write(f"// Size: {len(file_data)} bytes\n\n")
        f.write(f"#ifdef __has_attribute\n")
        f.write(f"#define alignas(x) __attribute__((aligned(x)))\n")
        f.write(f"#endif\n\n")
        f.write(f"const unsigned char {array_name}[] alignas(8) = {{\n  ")
        
        # Write array elements 12 per line for readability
        for i, hex_val in enumerate(hex_array):
            f.write(hex_val)
            if i < len(hex_array) - 1:
                f.write(", ")
                if (i + 1) % 12 == 0:
                    f.write("\n  ")
                    
        f.write("\n};\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(file_data)};\n\n")
        f.write("#endif // MODEL_H\n")

if __name__ == '__main__':
    if os.path.exists('model_quant.tflite'):
        tflite_to_c_array('model_quant.tflite', 'model.h')
        print(f"Successfully converted model_quant.tflite to model.h (Size: {os.path.getsize('model.h') / 1024 / 1024:.2f} MB)")
    else:
        print("Error: model_quant.tflite not found.")
