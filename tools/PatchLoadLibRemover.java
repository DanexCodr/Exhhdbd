package tools;

import org.objectweb.asm.*;

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
                return new MethodVisitor(Opcodes.ASM9, mv) {
                    @Override
                    public void visitMethodInsn(int opcode, String owner, String methodName, String methodDesc, boolean isInterface) {
                        // Detect System.loadLibrary(String)
                        if (opcode == Opcodes.INVOKESTATIC
                            && owner.equals("java/lang/System")
                            && methodName.equals("loadLibrary")
                            && methodDesc.equals("(Ljava/lang/String;)V")) {
                            // Skip this call (remove it) OR
                            // You can replace with System.load(...) if you want:
                            // super.visitLdcInsn("full path here");
                            // super.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/System", "load", "(Ljava/lang/String;)V", false);
                            // For now, just remove call by not calling super.visitMethodInsn
                            System.out.println("Removed call to System.loadLibrary");
                            // Do NOT call super.visitMethodInsn -> removes the call
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
