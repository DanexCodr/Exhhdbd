package tools;

import org.objectweb.asm.*;
import org.objectweb.asm.tree.*;

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

        // Use ClassNode to manipulate whole class
        ClassNode classNode = new ClassNode();
        cr.accept(classNode, ClassReader.EXPAND_FRAMES);

        for (MethodNode method : classNode.methods) {
            InsnList instructions = method.instructions;
            for (AbstractInsnNode insn = instructions.getFirst(); insn != null; ) {
                AbstractInsnNode next = insn.getNext();

                // Check for pattern: LDC (the library name) followed immediately by System.loadLibrary call
                if (insn.getOpcode() == Opcodes.LDC
                        && next != null
                        && next.getOpcode() == Opcodes.INVOKESTATIC) {

                    MethodInsnNode methodInsn = (MethodInsnNode) next;
                    if ("java/lang/System".equals(methodInsn.owner)
                            && "loadLibrary".equals(methodInsn.name)
                            && "(Ljava/lang/String;)V".equals(methodInsn.desc)) {

                        System.out.println("Removed call to System.loadLibrary");
                        System.out.println("Removed argument load: " + ((LdcInsnNode) insn).cst);

                        // Remove both the LDC and the call instructions
                        instructions.remove(insn);
                        instructions.remove(next);

                        // Continue from the instruction after removed call
                        insn = next.getNext();
                        continue;
                    }
                }

                insn = next;
            }
        }

        // Write modified class back (first pass)
        byte[] modifiedClass = cw.toByteArray();

        // Second pass to fully recompute frames and maxs
        ClassReader cr2 = new ClassReader(modifiedClass);
        ClassWriter cw2 = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        cr2.accept(cw2, ClassReader.EXPAND_FRAMES);
        byte[] fixedClass = cw2.toByteArray();

        // Write the fully fixed class back to file
        FileOutputStream fos = new FileOutputStream(classFile);
        fos.write(fixedClass);
        fos.close();

        System.out.println("Patched " + classFilePath + " successfully.");
    }
}
