OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5520369562225635) q[0];
rz(0.3861095309347213) q[0];
ry(1.486682226555432) q[1];
rz(-4.539992668028958e-05) q[1];
ry(-2.4944266182369232) q[2];
rz(3.140389038738403) q[2];
ry(-1.5706066016329012) q[3];
rz(-5.1529964373345124e-05) q[3];
ry(2.0419467885548457) q[4];
rz(-3.140991183099093) q[4];
ry(-1.5711812583170899) q[5];
rz(0.00012653015417979645) q[5];
ry(1.5708632496008565) q[6];
rz(3.1408145098886266) q[6];
ry(-1.5191573116844772) q[7];
rz(-2.6136344738477626) q[7];
ry(-2.620781933429454) q[8];
rz(1.580836764688306) q[8];
ry(-1.57076650364683) q[9];
rz(-1.4946399810754625) q[9];
ry(2.79116636338943) q[10];
rz(-1.563651659528559) q[10];
ry(1.5677905201677642) q[11];
rz(-0.7721696236783684) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.1271092902054596) q[0];
rz(1.9566177652928316) q[0];
ry(1.5710790861697426) q[1];
rz(2.1703516990920844) q[1];
ry(1.5716400819479523) q[2];
rz(-3.0874588181505636) q[2];
ry(-0.6208214195271415) q[3];
rz(1.5707997821592274) q[3];
ry(1.5701590111025536) q[4];
rz(-2.603538927850505) q[4];
ry(0.43088318418824656) q[5];
rz(-1.571490772538636) q[5];
ry(0.16521947108090898) q[6];
rz(1.651085735213254) q[6];
ry(-0.013229405567177697) q[7];
rz(-2.0986682356910666) q[7];
ry(0.0308223726234986) q[8];
rz(-2.36145761059737) q[8];
ry(3.1415644970689875) q[9];
rz(2.4481043540984104) q[9];
ry(-0.015623835629305837) q[10];
rz(-1.0108926810554313) q[10];
ry(3.1403272898356565) q[11];
rz(-2.3400714886633733) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.8000809327973153) q[0];
rz(-1.3567435014425973) q[0];
ry(0.06877645850132555) q[1];
rz(-2.3192927922086097) q[1];
ry(-0.7052753755484227) q[2];
rz(1.8096624795452314) q[2];
ry(0.6016826669091948) q[3];
rz(-1.5709189914461112) q[3];
ry(2.293566699515559) q[4];
rz(-1.5469562635062601) q[4];
ry(1.0048748672316634) q[5];
rz(1.8497297124625778) q[5];
ry(0.0004898261052916908) q[6];
rz(1.491277569783802) q[6];
ry(-1.7184365034444797) q[7];
rz(1.572439609464045) q[7];
ry(3.139205088379471) q[8];
rz(-0.7807777260572423) q[8];
ry(-0.00021368552819645004) q[9];
rz(-0.8011074536883972) q[9];
ry(0.0002285679892335474) q[10];
rz(2.573981089304642) q[10];
ry(-2.888526642490284) q[11];
rz(-1.5655825927026656) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.08392187846367405) q[0];
rz(-1.7840772280497799) q[0];
ry(3.140914658973681) q[1];
rz(-2.8382580382953133) q[1];
ry(3.140627771026287) q[2];
rz(2.2910143018062565) q[2];
ry(-2.9201419666114283) q[3];
rz(0.46218459820641034) q[3];
ry(-0.0007181483316953674) q[4];
rz(2.4052337050927077) q[4];
ry(3.140596825274043) q[5];
rz(-0.5909846177621386) q[5];
ry(1.6195239067840381) q[6];
rz(-3.141319413469986) q[6];
ry(1.595525493849304) q[7];
rz(0.7971099549245739) q[7];
ry(1.5723143084918778) q[8];
rz(2.259403051002469) q[8];
ry(-0.1999396867687606) q[9];
rz(1.5710246548750422) q[9];
ry(1.5705488940730234) q[10];
rz(1.7597869470806637) q[10];
ry(-0.2647222734932688) q[11];
rz(3.1389309111951142) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.023084655792103167) q[0];
rz(-1.5733478197002135) q[0];
ry(0.0019701205208633083) q[1];
rz(-0.361752998730632) q[1];
ry(3.1113244256260564) q[2];
rz(0.6881204210968488) q[2];
ry(4.418477521994646e-05) q[3];
rz(-0.4613036903279637) q[3];
ry(-1.427210291228663) q[4];
rz(2.00798969525324) q[4];
ry(3.1412646268569633) q[5];
rz(-1.5321587631259137) q[5];
ry(-1.5706075523493574) q[6];
rz(-0.9147925670227853) q[6];
ry(0.027295370649831337) q[7];
rz(0.5697045363254888) q[7];
ry(-1.9216135115160125) q[8];
rz(-2.4650354507223895) q[8];
ry(1.8667676524339312) q[9];
rz(0.8148128309473793) q[9];
ry(2.606048290828051) q[10];
rz(-2.3852574387471432) q[10];
ry(1.570822192802219) q[11];
rz(0.05657205250458458) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.0619593123975606) q[0];
rz(-1.5718717786079368) q[0];
ry(5.7530060578282164e-05) q[1];
rz(1.9761081867671804) q[1];
ry(-0.0004138298226390618) q[2];
rz(-1.264774905023814) q[2];
ry(-0.08038993118445875) q[3];
rz(-0.685387955645985) q[3];
ry(-0.00028721823725330603) q[4];
rz(1.3664311582873276) q[4];
ry(-0.0011928504563742948) q[5];
rz(-1.8886081505756378) q[5];
ry(0.00039802130508134097) q[6];
rz(2.1619672082268813) q[6];
ry(-1.7660089721207152e-05) q[7];
rz(0.2872211808665499) q[7];
ry(1.5710884058435333) q[8];
rz(1.5696799885280965) q[8];
ry(3.1414871939620364) q[9];
rz(-2.3280805335071286) q[9];
ry(1.5711611888543555) q[10];
rz(1.5701704025069567) q[10];
ry(-1.8051975819943777e-05) q[11];
rz(1.7702540043308037) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.4759531223644249) q[0];
rz(-1.5706029438807818) q[0];
ry(-3.133567220236337) q[1];
rz(-1.1751354980052913) q[1];
ry(-3.0717965499067854) q[2];
rz(-0.7381607059478316) q[2];
ry(-3.1415710770153846) q[3];
rz(0.7817179051537788) q[3];
ry(0.11621597607165501) q[4];
rz(-0.5134164839081838) q[4];
ry(3.141389726744649) q[5];
rz(1.9583233499520976) q[5];
ry(0.0011757858440342162) q[6];
rz(-1.389803230285164) q[6];
ry(1.1614981815312628) q[7];
rz(0.016066911838123452) q[7];
ry(-0.8876486700165261) q[8];
rz(3.1276730221766624) q[8];
ry(2.870124089046622) q[9];
rz(-0.7053369536384668) q[9];
ry(1.8862684916953147) q[10];
rz(-0.0014733015026753631) q[10];
ry(0.0001040264941529756) q[11];
rz(-0.11338762700405614) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.491290418226332) q[0];
rz(2.0592832162175836) q[0];
ry(1.5439868906056191) q[1];
rz(-1.3171287067346256) q[1];
ry(-2.988787132512173) q[2];
rz(-1.2232632885452102) q[2];
ry(-2.706050977952627) q[3];
rz(3.024014618778086) q[3];
ry(0.0004586997419231409) q[4];
rz(-2.2256975953084237) q[4];
ry(1.0897372900962299) q[5];
rz(0.40913713902004123) q[5];
ry(1.5546523785200292) q[6];
rz(0.9779390743811209) q[6];
ry(-1.5707585981203582) q[7];
rz(-2.9352052043060004) q[7];
ry(-1.9492266069362985) q[8];
rz(2.0418227015694326) q[8];
ry(2.7968120455113854) q[9];
rz(-0.7275907969698793) q[9];
ry(2.051208898277509) q[10];
rz(-1.0746344615873091) q[10];
ry(3.8300730428401014e-05) q[11];
rz(0.9266864733763311) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1415209805458844) q[0];
rz(-1.0822964583910775) q[0];
ry(0.008716787556500627) q[1];
rz(2.1537738391535477) q[1];
ry(3.141424844687568) q[2];
rz(0.4169773382456565) q[2];
ry(-1.5407497400193475) q[3];
rz(1.3722209983872649) q[3];
ry(-3.14155730928997) q[4];
rz(1.9262494455460482) q[4];
ry(1.574244055816445) q[5];
rz(-0.1754744111716926) q[5];
ry(3.139098377354535) q[6];
rz(2.5322575992019507) q[6];
ry(-3.129487794036401) q[7];
rz(0.02007280077664288) q[7];
ry(-3.1405517947041717) q[8];
rz(-2.0408490711417784) q[8];
ry(-1.5239730846742185) q[9];
rz(-2.113335003943564) q[9];
ry(0.0008443534109927597) q[10];
rz(0.6685002707323721) q[10];
ry(-3.1415896563395056) q[11];
rz(-1.1569244688744733) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.837872570435004) q[0];
rz(-1.5708234229320126) q[0];
ry(3.141190250658118) q[1];
rz(2.757130711822376) q[1];
ry(-0.0001356866604442786) q[2];
rz(2.870311107013154) q[2];
ry(3.0130236126161902) q[3];
rz(1.565418380939047) q[3];
ry(0.0006163414924387165) q[4];
rz(-0.9305185974098528) q[4];
ry(-1.0259844537473732) q[5];
rz(0.051850321191003114) q[5];
ry(3.1415562378594317) q[6];
rz(-1.7388839306296773) q[6];
ry(1.5833491320967636) q[7];
rz(-3.141154057910156) q[7];
ry(3.1414833099591086) q[8];
rz(0.6066383542942617) q[8];
ry(-2.1175829619455033) q[9];
rz(1.3455183706374676) q[9];
ry(-3.1415516762138385) q[10];
rz(0.20462979738757286) q[10];
ry(3.141510925132126) q[11];
rz(1.8375283430201774) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5613324646722146) q[0];
rz(2.709635700687916) q[0];
ry(3.123833361365293) q[1];
rz(0.08612604707399285) q[1];
ry(-3.137495753953398) q[2];
rz(-1.741824490563407) q[2];
ry(-3.135721175496012) q[3];
rz(-2.7285913704064977) q[3];
ry(-3.1411974151360718) q[4];
rz(1.9135556798706104) q[4];
ry(3.1407233079679404) q[5];
rz(2.276388894984522) q[5];
ry(-3.1361176039106584) q[6];
rz(-1.6935839413215785) q[6];
ry(1.5770695265384154) q[7];
rz(-0.0012050071673703757) q[7];
ry(3.1362850234320914) q[8];
rz(-1.5985044223254061) q[8];
ry(-0.0022671850734394096) q[9];
rz(1.7917854656438568) q[9];
ry(-3.1408332255861398) q[10];
rz(-2.5310305913360014) q[10];
ry(3.1415592135934545) q[11];
rz(1.5626970312246868) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1415850280412623) q[0];
rz(2.7027867073609544) q[0];
ry(-3.0105998151962017) q[1];
rz(-1.8342765578495426) q[1];
ry(1.4178790925181162) q[2];
rz(-0.27625646950412536) q[2];
ry(-3.1271979900568296) q[3];
rz(-1.1738086536053673) q[3];
ry(-1.570846235421302) q[4];
rz(1.602715819700385) q[4];
ry(-3.131940457294523) q[5];
rz(1.3121135498473508) q[5];
ry(-0.5317422602940809) q[6];
rz(1.5242288152130232) q[6];
ry(1.5694634499704305) q[7];
rz(1.5992866224125937) q[7];
ry(-0.9357248609402111) q[8];
rz(0.09444076683686298) q[8];
ry(0.025252607487295187) q[9];
rz(1.0264054967936431) q[9];
ry(-1.7210037096107875) q[10];
rz(0.03120482664694269) q[10];
ry(3.2698259309960065e-05) q[11];
rz(-1.4343665277115631) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.0981731521978118) q[0];
rz(1.5639632795189558) q[0];
ry(-1.5709028311908586) q[1];
rz(1.6236723970407556) q[1];
ry(-0.0001393535465874507) q[2];
rz(-0.5060379486035502) q[2];
ry(1.4967740152049938) q[3];
rz(3.100619037016177) q[3];
ry(-0.009899711896837358) q[4];
rz(-0.06817430690505634) q[4];
ry(-1.640968305913285) q[5];
rz(-3.096158678863814) q[5];
ry(-1.5702867049267288) q[6];
rz(-1.5355777743098136) q[6];
ry(-1.5087892830142005) q[7];
rz(-0.03942653197868279) q[7];
ry(3.123507884517694) q[8];
rz(-3.0351153911120865) q[8];
ry(1.5136982584465963) q[9];
rz(3.121072250406551) q[9];
ry(-3.1122363267717237) q[10];
rz(0.05742204751666944) q[10];
ry(-0.0004129410188229342) q[11];
rz(-0.6841239631776874) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.2778284069589425) q[0];
rz(1.5708148509808877) q[0];
ry(-1.570737060906721) q[1];
rz(1.6173033734845876) q[1];
ry(3.139768588616764) q[2];
rz(-1.278231233297452) q[2];
ry(-1.5708082908551484) q[3];
rz(1.9403497256121915) q[3];
ry(-0.06396427493641571) q[4];
rz(2.01477096215517) q[4];
ry(-1.5707867498661239) q[5];
rz(2.7487858745383824) q[5];
ry(0.06991428390505831) q[6];
rz(0.30340413373604846) q[6];
ry(-1.5708014388840816) q[7];
rz(-2.783330560705762) q[7];
ry(-0.09195986043417737) q[8];
rz(1.5660447237462094) q[8];
ry(-1.5707793727802843) q[9];
rz(-0.05592200394911995) q[9];
ry(-0.09349155116773021) q[10];
rz(1.5294217863116577) q[10];
ry(1.5707542839587734) q[11];
rz(-3.141578073910988) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5707746652304762) q[0];
rz(-0.01652966718780372) q[0];
ry(2.0153349323592806e-05) q[1];
rz(-0.04912026116830509) q[1];
ry(-4.982462691480877e-06) q[2];
rz(0.5225408944594401) q[2];
ry(-3.141575815857091) q[3];
rz(2.321208907935023) q[3];
ry(4.424810200368069e-06) q[4];
rz(-2.1398977512724118) q[4];
ry(-2.0091269897353993e-05) q[5];
rz(2.3350140483281554) q[5];
ry(3.141579811124562) q[6];
rz(0.9233235371334461) q[6];
ry(1.2991967138132209e-05) q[7];
rz(-0.5632793383835599) q[7];
ry(-3.141572252020421) q[8];
rz(0.22713110275578696) q[8];
ry(3.4209786332517895e-05) q[9];
rz(-2.627998356887564) q[9];
ry(-3.141574073169803) q[10];
rz(0.1865380242673204) q[10];
ry(-1.570809455283808) q[11];
rz(-2.2425141170702393e-06) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.021553918341910894) q[0];
rz(2.417464245573414) q[0];
ry(1.5698159935239486) q[1];
rz(-0.7424116374376667) q[1];
ry(1.4860726827068718) q[2];
rz(-2.3131977282063056) q[2];
ry(-3.1389570364282102) q[3];
rz(1.2092218480541506) q[3];
ry(0.023039701210648047) q[4];
rz(2.562781449550141) q[4];
ry(0.0006854112404525651) q[5];
rz(2.0277789013677263) q[5];
ry(-0.05567185603829585) q[6];
rz(1.8182742812453616) q[6];
ry(0.0028873071812514084) q[7];
rz(1.0334812530950384) q[7];
ry(-0.11421572547463638) q[8];
rz(0.6289723017943096) q[8];
ry(-0.0007332107535238145) q[9];
rz(0.3708519148781209) q[9];
ry(-0.10872164319567089) q[10];
rz(0.6316236670386223) q[10];
ry(-1.5694473115060754) q[11];
rz(-2.5893774275713226) q[11];