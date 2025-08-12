package tools;

import org.objectweb.asm.*;
import org.objectweb.asm.commons.AdviceAdapter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class PatchLoadLibRemover {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: java tools.PatchLoadLibRemover <path-to-OnnxRuntime.class>");
            System.exit(1);
        }

        String classFilePath = args[0];
        File classFile = new File(classFilePath);
        if (!classFile.exists()) {
            System.err.println("Class file not found: " + classFilePath);
            System.exit(1);
        }

        // Read class bytes
        FileInputStream fis = new FileInputStream(classFile);
        byte[] classBytes = new byte[(int) classFile.length()];
        fis.read(classBytes);
        fis.close();

        ClassReader cr = new ClassReader(classBytes);
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);

        ClassVisitor cv = new ClassVisitor(Opcodes.ASM9, cw) {
            @Override
            public MethodVisitor visitMethod(int access, String name, String desc,
                                             String signature, String[] exceptions) {
                MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);

                return new AdviceAdapter(Opcodes.ASM9, mv, access, name, desc) {

                    private boolean skipNextLdc = false;

                    @Override
                    public void visitLdcInsn(Object value) {
                        if (skipNextLdc) {
                            // Skip this LDC instruction (argument to loadLibrary)
                            skipNextLdc = false;
                            System.out.println("Removed argument load: " + value);
                        } else {
                            super.visitLdcInsn(value);
                        }
                    }

                    @Override
                    public void visitMethodInsn(int opcode, String owner, String methodName, String methodDesc, boolean isInterface) {
                        if (opcode == Opcodes.INVOKESTATIC
                            && owner.equals("java/lang/System")
                            && methodName.equals("loadLibrary")
                            && methodDesc.equals("(Ljava/lang/String;)V")) {
                            // Remove the call + the argument (previous LDC)
                            skipNextLdc = true; // flag to skip previous LDC
                            System.out.println("Removed call to System.loadLibrary");
                            // Do NOT call super.visitMethodInsn, so call is removed
                        } else {
                            super.visitMethodInsn(opcode, owner, methodName, methodDesc, isInterface);
                        }
                    }
                };
            }
        };

        cr.accept(cv, 0);

        byte[] modifiedClass = cw.toByteArray();

        // Write back the patched class file
        FileOutputStream fos = new FileOutputStream(classFile);
        fos.write(modifiedClass);
        fos.close();

        System.out.println("Patched " + classFilePath + " successfully.");
    }
}
