OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.0776381264409132) q[0];
rz(-0.737290897886667) q[0];
ry(1.8349447689339637) q[1];
rz(0.054328674396434275) q[1];
ry(-1.7809979485248535) q[2];
rz(2.4473172425021628) q[2];
ry(-1.330333870369033) q[3];
rz(-2.463273694582381) q[3];
ry(0.001912212869705576) q[4];
rz(-2.8061128375502493) q[4];
ry(0.001378288907536884) q[5];
rz(-0.26703126625728846) q[5];
ry(2.421999102653499) q[6];
rz(2.4697155770990045) q[6];
ry(-1.5256921545138231) q[7];
rz(-0.320851602833207) q[7];
ry(-2.4815284718228034) q[8];
rz(-2.82542056649676) q[8];
ry(-1.058049631516348) q[9];
rz(-0.7923663381415753) q[9];
ry(-0.9807483050177279) q[10];
rz(2.0281163294472906) q[10];
ry(-0.6977120007663622) q[11];
rz(1.2647007368982721) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.1403816492938317) q[0];
rz(3.004373826892453) q[0];
ry(-2.0821088106960044) q[1];
rz(0.7872204236062601) q[1];
ry(1.0165333405338068) q[2];
rz(-2.0648853308987443) q[2];
ry(-2.4659925103602136) q[3];
rz(-2.4477085753219057) q[3];
ry(-0.0023907283670461287) q[4];
rz(1.0332238347255691) q[4];
ry(3.139847810794241) q[5];
rz(-0.04103710166781927) q[5];
ry(-2.877256476288818) q[6];
rz(2.2610238963231013) q[6];
ry(1.2846252077538756) q[7];
rz(-2.8583249803661066) q[7];
ry(-1.4944648392137538) q[8];
rz(0.430746986466267) q[8];
ry(-2.223996854349349) q[9];
rz(2.646934958763659) q[9];
ry(-2.812124184312333) q[10];
rz(0.07687942672327806) q[10];
ry(2.578172767361004) q[11];
rz(-1.2724911076270649) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.382669019948921) q[0];
rz(-2.300957688457627) q[0];
ry(1.4509303063126087) q[1];
rz(2.124030340864368) q[1];
ry(1.0467486699728124) q[2];
rz(-0.7817452927933521) q[2];
ry(2.00077032701712) q[3];
rz(1.289274582625136) q[3];
ry(-1.5549842788201735) q[4];
rz(0.5438075961430338) q[4];
ry(1.5902715234866291) q[5];
rz(-2.6800904093595777) q[5];
ry(-0.8116855897536636) q[6];
rz(0.8751462839256083) q[6];
ry(-2.2042280453104492) q[7];
rz(-1.4079704822916463) q[7];
ry(2.377018308530999) q[8];
rz(-2.0595367061566865) q[8];
ry(-1.5437764151827187) q[9];
rz(-1.8542133274463084) q[9];
ry(0.3376313497367507) q[10];
rz(-0.16933544680549395) q[10];
ry(1.3612775999719826) q[11];
rz(3.012753477096461) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.43921304739696243) q[0];
rz(1.6851366679733761) q[0];
ry(0.4345813344307712) q[1];
rz(1.7723035229720836) q[1];
ry(-0.004390742194332409) q[2];
rz(-0.009395868417058615) q[2];
ry(3.1338807150481807) q[3];
rz(1.767874389503878) q[3];
ry(0.0006157828634165098) q[4];
rz(2.5723774143167484) q[4];
ry(-0.0025292834204319733) q[5];
rz(2.667041905436561) q[5];
ry(-2.510718047278545) q[6];
rz(-1.0117334767206396) q[6];
ry(-2.8764089417845256) q[7];
rz(1.8839878284806773) q[7];
ry(2.441100133850685) q[8];
rz(-0.45325244645195634) q[8];
ry(1.0319920181355875) q[9];
rz(2.685972583068701) q[9];
ry(-1.557598123228776) q[10];
rz(-0.4791542673952218) q[10];
ry(1.511868192104129) q[11];
rz(-0.9022935049462718) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.20537719544518396) q[0];
rz(-1.436616358456543) q[0];
ry(-0.7085462665696356) q[1];
rz(2.0553928034675453) q[1];
ry(-0.9633284327575753) q[2];
rz(-2.846650906815384) q[2];
ry(-0.1648055075516437) q[3];
rz(-0.2199913393210269) q[3];
ry(1.5666247661383013) q[4];
rz(-0.7674173534141051) q[4];
ry(-1.544359097884727) q[5];
rz(1.0378325071802637) q[5];
ry(2.733334253439063) q[6];
rz(-0.7378170628160028) q[6];
ry(1.6118066472924264) q[7];
rz(-0.37113227512130736) q[7];
ry(1.9748318184632376) q[8];
rz(0.3417917289999126) q[8];
ry(0.6137017819346808) q[9];
rz(-0.9090533214367991) q[9];
ry(0.3952219453239305) q[10];
rz(2.6392747057419492) q[10];
ry(-0.8971439705513387) q[11];
rz(-0.5062228758650296) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.0721888832649924) q[0];
rz(-1.0630612215133128) q[0];
ry(-1.4707027350627024) q[1];
rz(-2.63420415872874) q[1];
ry(2.6377624966118582) q[2];
rz(-1.4197647747387396) q[2];
ry(1.1349039863364845) q[3];
rz(-0.2552293579848177) q[3];
ry(-1.5607991217528152) q[4];
rz(-0.9350367204587052) q[4];
ry(1.5610712776777893) q[5];
rz(0.7567578432480611) q[5];
ry(2.896436392482127) q[6];
rz(2.346212408881316) q[6];
ry(-2.581282563397119) q[7];
rz(-1.081504537790945) q[7];
ry(-2.4854037297439686) q[8];
rz(-0.9308586156754083) q[8];
ry(2.7977183052400045) q[9];
rz(1.7861814337844955) q[9];
ry(-1.496545755456046) q[10];
rz(0.7340585283987521) q[10];
ry(0.39016989949698344) q[11];
rz(2.74658203171179) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.6545152247032373) q[0];
rz(-0.28282050969414874) q[0];
ry(-0.5212667063259532) q[1];
rz(-2.6863755893647157) q[1];
ry(-1.5683750970513488) q[2];
rz(1.2301344309065787e-05) q[2];
ry(-1.5836292705481139) q[3];
rz(3.139382959175183) q[3];
ry(-3.140659656779108) q[4];
rz(-1.02598804320499) q[4];
ry(-3.139291761895861) q[5];
rz(1.0727445843881451) q[5];
ry(2.5120599405206163) q[6];
rz(-2.5392033265708998) q[6];
ry(0.5219356694883022) q[7];
rz(3.017254104971421) q[7];
ry(-1.4759943002178892) q[8];
rz(-2.459354427659907) q[8];
ry(1.3140406628915846) q[9];
rz(2.326583441264085) q[9];
ry(-0.6747349699820635) q[10];
rz(2.7350655971447124) q[10];
ry(1.5348148763155463) q[11];
rz(2.065365431338188) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.027343404881848273) q[0];
rz(1.058313314120655) q[0];
ry(-1.9174748586992472) q[1];
rz(1.865927305773992) q[1];
ry(-1.5504606637995426) q[2];
rz(2.2555533418461704) q[2];
ry(-1.5667216594442102) q[3];
rz(-0.8129382570588003) q[3];
ry(0.0028724604034588946) q[4];
rz(2.985229304659898) q[4];
ry(-3.1386648249705367) q[5];
rz(0.8556929299600088) q[5];
ry(2.0857391125174165) q[6];
rz(-1.033797916464696) q[6];
ry(1.1125861472231635) q[7];
rz(0.6027010910971671) q[7];
ry(-1.5519439075034476) q[8];
rz(-0.4834301370525764) q[8];
ry(2.3709843896071088) q[9];
rz(2.356835857037286) q[9];
ry(-0.7919058304994414) q[10];
rz(-0.004182810383816182) q[10];
ry(0.6831552620127859) q[11];
rz(0.7759349513337234) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.2218565298680168) q[0];
rz(-2.0460661459235983) q[0];
ry(-1.5243843132669654) q[1];
rz(-0.030797754507652535) q[1];
ry(-0.7985070525623789) q[2];
rz(1.927090374950173) q[2];
ry(1.8150304519438976) q[3];
rz(2.920083551626976) q[3];
ry(-0.7690058531735833) q[4];
rz(1.510011288736898) q[4];
ry(3.0533637151735853) q[5];
rz(-2.7334618796112657) q[5];
ry(0.4236992382801174) q[6];
rz(3.0960369541791333) q[6];
ry(0.2278400190797143) q[7];
rz(0.6988555957237937) q[7];
ry(1.6515688584559562) q[8];
rz(2.8557201165630954) q[8];
ry(2.224902624800756) q[9];
rz(-1.0336627203288167) q[9];
ry(-1.1820235636554497) q[10];
rz(-3.1201370116866407) q[10];
ry(1.0899055440081484) q[11];
rz(0.44694890172420193) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.536142372570601) q[0];
rz(2.1449923350197393) q[0];
ry(0.20072543410430208) q[1];
rz(-2.4331057362623567) q[1];
ry(-0.013506993926695188) q[2];
rz(1.6323403386415691) q[2];
ry(3.1129945733193027) q[3];
rz(0.5013963952545675) q[3];
ry(-0.005222888212472299) q[4];
rz(-0.007373042184775081) q[4];
ry(3.126189520218348) q[5];
rz(-2.5973974201709935) q[5];
ry(1.1294349778053068) q[6];
rz(0.18404349891426453) q[6];
ry(1.5230622475114624) q[7];
rz(1.9140839045311253) q[7];
ry(-3.1280671532374478) q[8];
rz(1.097335815254386) q[8];
ry(0.47815251514709295) q[9];
rz(-1.4856203783745188) q[9];
ry(0.741729344197929) q[10];
rz(0.3882283229923764) q[10];
ry(-1.688844073506582) q[11];
rz(-0.6902774910059365) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.667654737139216) q[0];
rz(-0.7768508521680602) q[0];
ry(0.890200687432135) q[1];
rz(-1.586120814552011) q[1];
ry(-1.9142935916419654) q[2];
rz(1.8758738113572255) q[2];
ry(1.6513343669433453) q[3];
rz(-0.3868774173583329) q[3];
ry(-2.3575386854182043) q[4];
rz(-1.3738713265560503) q[4];
ry(-0.8402863344637292) q[5];
rz(1.0025332037976908) q[5];
ry(-1.4844326001352397) q[6];
rz(1.223342005262979) q[6];
ry(-2.235712492047032) q[7];
rz(-0.7142756663649665) q[7];
ry(1.8849738632917745) q[8];
rz(0.9062228934428128) q[8];
ry(0.7511425087053221) q[9];
rz(1.5117377285999014) q[9];
ry(-1.7372122473095672) q[10];
rz(-0.9497338481260285) q[10];
ry(-2.169112484352417) q[11];
rz(-3.1117467503552096) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.4729533302624023) q[0];
rz(0.5104473442815409) q[0];
ry(-3.1140815712828256) q[1];
rz(1.6640985837636946) q[1];
ry(-3.1204097179273615) q[2];
rz(2.996000180455433) q[2];
ry(0.011240091891034254) q[3];
rz(2.4284192064415024) q[3];
ry(-0.008962841297806924) q[4];
rz(1.0893195102743831) q[4];
ry(-0.00016084881656297512) q[5];
rz(0.20496078736084392) q[5];
ry(2.7511895159861735) q[6];
rz(-1.5760572382385987) q[6];
ry(0.6307404296724792) q[7];
rz(2.195565249979712) q[7];
ry(0.49913921406086015) q[8];
rz(-1.630445262468601) q[8];
ry(-2.817239062062035) q[9];
rz(3.0900744286661643) q[9];
ry(2.151027267873583) q[10];
rz(2.8024641756951807) q[10];
ry(2.3816087301936792) q[11];
rz(-1.9199183449926482) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.7118955162065888) q[0];
rz(2.425474884907271) q[0];
ry(0.8788039880184613) q[1];
rz(2.48847079968055) q[1];
ry(-0.865414699201521) q[2];
rz(1.6807963004071118) q[2];
ry(-1.8831514082049505) q[3];
rz(-1.7587886599062705) q[3];
ry(1.4683207242087053) q[4];
rz(-0.8088047245639564) q[4];
ry(1.8396458094689734) q[5];
rz(-0.037104718879171786) q[5];
ry(2.8314806549745906) q[6];
rz(-2.908484110386189) q[6];
ry(1.7605816282086366) q[7];
rz(0.4225383298318886) q[7];
ry(2.2279770004631425) q[8];
rz(-0.7304872842381196) q[8];
ry(1.303720562945948) q[9];
rz(-1.093322901081099) q[9];
ry(-2.6183310803275117) q[10];
rz(2.4896867573754315) q[10];
ry(-2.2906671883209815) q[11];
rz(-2.5524393003961436) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.0944151226012897) q[0];
rz(2.9720596280244727) q[0];
ry(-1.8663723109666168) q[1];
rz(-2.8689612671569216) q[1];
ry(3.116849338751552) q[2];
rz(1.4694190963523353) q[2];
ry(3.1246873706348093) q[3];
rz(-0.9310584524247227) q[3];
ry(-0.004076246071160191) q[4];
rz(0.6831028482202609) q[4];
ry(3.136672547595093) q[5];
rz(-0.7060277645912959) q[5];
ry(2.856722892239827) q[6];
rz(-0.23041748119864894) q[6];
ry(0.6039898286548111) q[7];
rz(2.3683426478290275) q[7];
ry(-1.2379343937043246) q[8];
rz(-0.8313755275181567) q[8];
ry(-2.489115511818059) q[9];
rz(0.45671454745790946) q[9];
ry(2.7988259087511884) q[10];
rz(3.057829953620787) q[10];
ry(0.8733594724104129) q[11];
rz(-2.307623730397059) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.9611691604844301) q[0];
rz(-2.8521083118022372) q[0];
ry(-0.18729873629731097) q[1];
rz(0.19564489517748762) q[1];
ry(-2.7275413917323745) q[2];
rz(-1.9117110283981702) q[2];
ry(-0.2188885871288695) q[3];
rz(-1.9428716002764868) q[3];
ry(-0.2321514464920611) q[4];
rz(0.3272485696033617) q[4];
ry(3.0076743864982007) q[5];
rz(-1.9285722015884073) q[5];
ry(0.32155574350261507) q[6];
rz(-0.9363698029900654) q[6];
ry(0.029189698206500234) q[7];
rz(3.0119468937785343) q[7];
ry(-2.145802762052576) q[8];
rz(-2.80441133112406) q[8];
ry(-0.21624228123263034) q[9];
rz(0.3805191694552067) q[9];
ry(0.519377865607569) q[10];
rz(1.3671568061438162) q[10];
ry(-0.6344205119021895) q[11];
rz(-1.2461784898117703) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.3863472035450664) q[0];
rz(-1.1735917300939827) q[0];
ry(-0.1850784195961789) q[1];
rz(2.1786421747283016) q[1];
ry(0.018366517173023806) q[2];
rz(-2.521047469933971) q[2];
ry(-3.1382960568841436) q[3];
rz(-1.8956210017784276) q[3];
ry(0.0037227782854773357) q[4];
rz(1.4445922093619303) q[4];
ry(3.1409093879939607) q[5];
rz(2.080713639501462) q[5];
ry(2.575230675467129) q[6];
rz(-0.9482155204370172) q[6];
ry(1.7654611683842738) q[7];
rz(-2.8040838169980358) q[7];
ry(-1.114113232603751) q[8];
rz(2.4785692372297903) q[8];
ry(-0.7423749867632462) q[9];
rz(0.19243703680256008) q[9];
ry(1.2536160051723821) q[10];
rz(2.083697353016014) q[10];
ry(-1.469356154012852) q[11];
rz(-1.4168342133792027) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.20797687721196303) q[0];
rz(-2.0078498119224486) q[0];
ry(-1.5119047137246666) q[1];
rz(1.3560279737502523) q[1];
ry(-2.531505722447867) q[2];
rz(2.4135093536707966) q[2];
ry(-2.729963272246293) q[3];
rz(1.8466266924242725) q[3];
ry(1.5295534432658737) q[4];
rz(-2.8727640418976663) q[4];
ry(0.38422915206122266) q[5];
rz(1.4211874149311092) q[5];
ry(0.8777305279125116) q[6];
rz(2.180810691574414) q[6];
ry(0.6857860174061772) q[7];
rz(1.050782080451195) q[7];
ry(-1.6430252662578768) q[8];
rz(-1.961472428041838) q[8];
ry(1.5527134873863941) q[9];
rz(-0.31841307777115807) q[9];
ry(-2.7375025097534387) q[10];
rz(0.09114215314700602) q[10];
ry(-1.2874196138896012) q[11];
rz(-3.016810144735517) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.5731810417356633) q[0];
rz(-2.713620780962255) q[0];
ry(-1.57374825213071) q[1];
rz(-1.0848926398439307) q[1];
ry(0.8597183498684196) q[2];
rz(-0.1425073284291143) q[2];
ry(-1.3634946632947555) q[3];
rz(-2.9960057575184487) q[3];
ry(-3.1386086631867447) q[4];
rz(0.3699667526183697) q[4];
ry(0.010206183603444984) q[5];
rz(1.6186046934151141) q[5];
ry(-2.2516772252280592) q[6];
rz(-2.2317786160388478) q[6];
ry(-2.414608364081222) q[7];
rz(2.0149108564617126) q[7];
ry(0.24629546436865205) q[8];
rz(0.06336210661703315) q[8];
ry(0.31739172283800343) q[9];
rz(-1.72607720719175) q[9];
ry(1.216459363290035) q[10];
rz(2.871542684772356) q[10];
ry(1.6141371892025371) q[11];
rz(-0.897290689942655) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.6277724984087176) q[0];
rz(-1.6865111104822978) q[0];
ry(-0.6453362750083445) q[1];
rz(-0.2950477583011802) q[1];
ry(-2.0919599599521304) q[2];
rz(1.0867080231238881) q[2];
ry(-1.9615346971330618) q[3];
rz(-1.117790945599372) q[3];
ry(3.141016900289743) q[4];
rz(2.993007589087621) q[4];
ry(0.009352043287781164) q[5];
rz(-0.7621748142128668) q[5];
ry(1.5456404521036624) q[6];
rz(1.3457415026919843) q[6];
ry(-2.0694964195126175) q[7];
rz(-1.2431236772922964) q[7];
ry(-0.5251482260566513) q[8];
rz(0.28410657022966124) q[8];
ry(-2.9090092756606087) q[9];
rz(0.7643029067366428) q[9];
ry(0.6952540247115673) q[10];
rz(0.9628835266775281) q[10];
ry(-1.1133784041414456) q[11];
rz(0.900995257386403) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.06349091107669302) q[0];
rz(-1.5538536223748756) q[0];
ry(-0.4920357018327497) q[1];
rz(1.1692750098254083) q[1];
ry(-1.5652474450709413) q[2];
rz(-0.3555669219728826) q[2];
ry(-1.9182026511781594) q[3];
rz(-0.4373613460701966) q[3];
ry(3.077254660137572) q[4];
rz(2.470744279121538) q[4];
ry(-3.126906184940776) q[5];
rz(-2.618000633888148) q[5];
ry(-2.447241500262757) q[6];
rz(0.43035511944479504) q[6];
ry(0.8079507516373065) q[7];
rz(-2.011223009552806) q[7];
ry(0.19355021140801085) q[8];
rz(-1.8793263679996866) q[8];
ry(2.780488059217149) q[9];
rz(2.5684329314177488) q[9];
ry(-1.4876208117980922) q[10];
rz(-0.6682333313596001) q[10];
ry(-1.3754283816241004) q[11];
rz(-2.1667017143140774) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.2092605053475292) q[0];
rz(-1.735025738398187) q[0];
ry(2.2843838236020915) q[1];
rz(1.3869215153732186) q[1];
ry(1.7727575474291175) q[2];
rz(-1.4677240975420303) q[2];
ry(-3.0123806125346984) q[3];
rz(-0.4898073171273243) q[3];
ry(-0.022110482105111212) q[4];
rz(-0.261891589937963) q[4];
ry(-0.010655083643548788) q[5];
rz(0.26175539931441016) q[5];
ry(2.280782740759716) q[6];
rz(2.5332623610066687) q[6];
ry(-0.03706257126980983) q[7];
rz(-3.022858182836696) q[7];
ry(1.34025024618549) q[8];
rz(-1.7120095059687337) q[8];
ry(1.55840398974131) q[9];
rz(-2.5312296097848535) q[9];
ry(0.1906452423126167) q[10];
rz(-1.2882062699000065) q[10];
ry(-0.9695293677898142) q[11];
rz(1.3440812730739875) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.885280729989396) q[0];
rz(-2.806110434593382) q[0];
ry(0.07678987409324378) q[1];
rz(0.9730748275256205) q[1];
ry(-1.7451068690646743) q[2];
rz(1.2674979801581248) q[2];
ry(-1.368425520390667) q[3];
rz(-1.0495725246897953) q[3];
ry(0.015126073649306626) q[4];
rz(0.7022130575618766) q[4];
ry(-0.002379227933542082) q[5];
rz(0.5678263631591921) q[5];
ry(-0.11835761919777289) q[6];
rz(-1.7862372791347167) q[6];
ry(2.8950523502595953) q[7];
rz(1.611675196437182) q[7];
ry(1.9458169086414447) q[8];
rz(-2.9338957884413275) q[8];
ry(-1.4784066487350174) q[9];
rz(-2.3235617451774138) q[9];
ry(-0.9387568414720641) q[10];
rz(-2.7707305914632525) q[10];
ry(0.525980717351958) q[11];
rz(0.8538501319778052) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.4574846111174122) q[0];
rz(2.370125131346963) q[0];
ry(1.305504663540984) q[1];
rz(-0.8399028299841954) q[1];
ry(1.64605668629104) q[2];
rz(-2.5720376776996927) q[2];
ry(0.1349670075824498) q[3];
rz(-2.619349080062356) q[3];
ry(0.001049672882524558) q[4];
rz(0.8424840523486044) q[4];
ry(3.1386136673395884) q[5];
rz(1.0810052051487462) q[5];
ry(-1.0014811898838278) q[6];
rz(-1.5026495137363156) q[6];
ry(-1.7471629726991065) q[7];
rz(-1.9027265827602955) q[7];
ry(-2.544225854040453) q[8];
rz(-1.9218218487639884) q[8];
ry(-1.2662463546884286) q[9];
rz(-0.11078797485108248) q[9];
ry(0.750901503919969) q[10];
rz(-2.327929200323809) q[10];
ry(1.814410244951013) q[11];
rz(-0.8334862800983011) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1105404150274265) q[0];
rz(-2.7653087951423077) q[0];
ry(0.1009728074478149) q[1];
rz(-2.828946225466528) q[1];
ry(-2.9544431685112817) q[2];
rz(0.52271329117911) q[2];
ry(-2.699697605651299) q[3];
rz(1.1224554327640492) q[3];
ry(-0.006809047011885229) q[4];
rz(-1.893669730753599) q[4];
ry(2.484169459509636) q[5];
rz(1.3662432661641948) q[5];
ry(-1.8682312765055402) q[6];
rz(2.936325341851731) q[6];
ry(-1.3952144881495483) q[7];
rz(-2.8051449554134287) q[7];
ry(-2.569768155106034) q[8];
rz(-2.1845043064354503) q[8];
ry(-0.6639193007403179) q[9];
rz(-0.810040577151284) q[9];
ry(-0.7773679613534991) q[10];
rz(-3.1326621078226955) q[10];
ry(-0.6012944056628758) q[11];
rz(2.8352332128849613) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5556837737909124) q[0];
rz(0.20125825900351027) q[0];
ry(2.5853862620115433) q[1];
rz(-1.4319826980201429) q[1];
ry(1.9917805445849726) q[2];
rz(-1.6382493912895686) q[2];
ry(-0.33880087850310225) q[3];
rz(-1.5289721454782708) q[3];
ry(3.136485112753702) q[4];
rz(-2.390452287048938) q[4];
ry(3.136262526918142) q[5];
rz(1.4060576502914301) q[5];
ry(-3.113504507193069) q[6];
rz(-0.481042817787709) q[6];
ry(-3.1302577308945967) q[7];
rz(-2.7178904596245634) q[7];
ry(-1.9967873527606477) q[8];
rz(-2.073682571257258) q[8];
ry(-2.571898021282537) q[9];
rz(2.4045024474561103) q[9];
ry(1.2894740473758066) q[10];
rz(-0.17071507884834425) q[10];
ry(-1.5894788528732307) q[11];
rz(-2.306451943543391) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.05498714682506977) q[0];
rz(2.969375018520264) q[0];
ry(2.004013409614672) q[1];
rz(1.9392473595257202) q[1];
ry(-1.603527128786972) q[2];
rz(-2.4638838849389657) q[2];
ry(1.5482907439292992) q[3];
rz(-0.2036659779344676) q[3];
ry(2.4972655786685922) q[4];
rz(3.1398939384386675) q[4];
ry(0.4406679774843823) q[5];
rz(3.111083682316518) q[5];
ry(-1.5099190512650376) q[6];
rz(1.8918257063102626) q[6];
ry(1.760133087853803) q[7];
rz(1.8152287024658613) q[7];
ry(-0.49941709037045506) q[8];
rz(-0.09404104199389529) q[8];
ry(1.834602633484329) q[9];
rz(0.06220076255769946) q[9];
ry(0.48311229010753487) q[10];
rz(-1.596749580102149) q[10];
ry(0.6514173918275927) q[11];
rz(-0.376581273312278) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.02640242575882415) q[0];
rz(-2.438565407274309) q[0];
ry(2.0707443652562634) q[1];
rz(2.3384406929595087) q[1];
ry(0.012124781265232976) q[2];
rz(2.438760086129295) q[2];
ry(0.04757513093042576) q[3];
rz(-0.11055242853315511) q[3];
ry(-1.5670573280525515) q[4];
rz(1.7579851301142622) q[4];
ry(1.5580773536188186) q[5];
rz(-0.023708264072283214) q[5];
ry(-1.6107586324824055) q[6];
rz(-2.735441259553322) q[6];
ry(-1.622606420649947) q[7];
rz(0.0020354228936279033) q[7];
ry(2.3804914412290303) q[8];
rz(1.5309824900063829) q[8];
ry(1.4599644534994214) q[9];
rz(-0.6934685105969376) q[9];
ry(-1.1239951162884205) q[10];
rz(-0.9386629662618308) q[10];
ry(-0.11672456451539216) q[11];
rz(1.2506708936173823) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.8097469610700947) q[0];
rz(0.2928975162987432) q[0];
ry(-2.2471536369489176) q[1];
rz(-2.269351420618458) q[1];
ry(-1.3984745108381642) q[2];
rz(-1.5588644686482132) q[2];
ry(-1.1892486654487247) q[3];
rz(-1.5214496395497001) q[3];
ry(2.991559222249261) q[4];
rz(0.19242700599254173) q[4];
ry(3.1330096317938336) q[5];
rz(-1.5433510310157557) q[5];
ry(0.12185410619757421) q[6];
rz(1.1715969740757448) q[6];
ry(1.3696794418711757) q[7];
rz(-1.5150029134580332) q[7];
ry(-1.1996903736777291) q[8];
rz(1.766543447070033) q[8];
ry(-0.5778487405452565) q[9];
rz(-1.7197601802319353) q[9];
ry(-1.9588339614491794) q[10];
rz(-1.954356566798803) q[10];
ry(-1.305985280214262) q[11];
rz(2.6314601956293835) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.6353712672665253) q[0];
rz(-2.571303628270248) q[0];
ry(1.6003957663333386) q[1];
rz(-2.57672503752876) q[1];
ry(1.4607937182062856) q[2];
rz(1.5798134925454947) q[2];
ry(-1.9460134607736082) q[3];
rz(1.5555232755571433) q[3];
ry(-1.5691323136048068) q[4];
rz(-1.4210178586604505) q[4];
ry(-0.007412567616206012) q[5];
rz(-0.32529523167683055) q[5];
ry(1.484819893519842) q[6];
rz(-0.2968297235187952) q[6];
ry(-1.494138067616104) q[7];
rz(0.44778033107665655) q[7];
ry(2.1186639515490873) q[8];
rz(3.078468139039721) q[8];
ry(2.6349342579232076) q[9];
rz(1.9557301691066638) q[9];
ry(1.5384878408822313) q[10];
rz(2.3218411717501337) q[10];
ry(-1.876989750449787) q[11];
rz(-1.5835444862642136) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.028790413428017914) q[0];
rz(0.8345797217386592) q[0];
ry(-3.1072056437592863) q[1];
rz(-2.2657251576091992) q[1];
ry(1.5659402809661522) q[2];
rz(0.001968746921579445) q[2];
ry(1.5676236275494961) q[3];
rz(-2.053822537771009) q[3];
ry(-3.1414767664969996) q[4];
rz(-2.9912311460576184) q[4];
ry(3.1140161547555785) q[5];
rz(-1.8544899926626701) q[5];
ry(-0.003981474949990092) q[6];
rz(0.24263594520478102) q[6];
ry(3.112622144799407) q[7];
rz(-1.8387671505170289) q[7];
ry(0.49685893275354953) q[8];
rz(0.005993122301745329) q[8];
ry(1.5366766905733016) q[9];
rz(2.0244876219607573) q[9];
ry(-1.5044504600122084) q[10];
rz(-2.751668585347963) q[10];
ry(1.8079908093504664) q[11];
rz(-0.7371098668262198) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.6730302242738873) q[0];
rz(-1.5390160233724108) q[0];
ry(0.09266229527064507) q[1];
rz(1.2553067585840978) q[1];
ry(-1.5737349523557334) q[2];
rz(2.4901322857130137) q[2];
ry(-3.141096685618921) q[3];
rz(2.6584054145280853) q[3];
ry(-1.5160318696875825) q[4];
rz(-2.24537685263295) q[4];
ry(-1.5698447022883173) q[5];
rz(0.0352715298787567) q[5];
ry(-0.0067074736677597) q[6];
rz(1.6219369284125111) q[6];
ry(3.1171212309519) q[7];
rz(2.4287341308112915) q[7];
ry(-1.64257366856146) q[8];
rz(0.9595281333252634) q[8];
ry(2.490869942317058) q[9];
rz(2.558618774184204) q[9];
ry(-0.8547552495331272) q[10];
rz(2.1619763959657137) q[10];
ry(1.8359956999424938) q[11];
rz(2.703297622650226) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5714097560440883) q[0];
rz(-2.099098126685898) q[0];
ry(1.5703308269409364) q[1];
rz(-2.0215671321550595) q[1];
ry(1.6599720001251486) q[2];
rz(-0.18622108983121866) q[2];
ry(-1.5751854225299908) q[3];
rz(2.7850658676394247) q[3];
ry(-3.1396717871319533) q[4];
rz(0.6312309674225398) q[4];
ry(3.1376657294856702) q[5];
rz(1.5658431054526414) q[5];
ry(1.5739075793375754) q[6];
rz(-3.0403593765951142) q[6];
ry(1.5763582473068363) q[7];
rz(-1.3858413622926937) q[7];
ry(-2.8666038004954273) q[8];
rz(-1.263733041474607) q[8];
ry(1.8675590701971967) q[9];
rz(-2.0234072637277887) q[9];
ry(-0.4443994728433678) q[10];
rz(-0.7940646409379883) q[10];
ry(2.602963061500132) q[11];
rz(1.9331225922522233) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.9538469592411647) q[0];
rz(-2.9905769587313165) q[0];
ry(-0.14535852532032012) q[1];
rz(-2.003214872391769) q[1];
ry(-1.4800738457792326) q[2];
rz(2.2443435562617924) q[2];
ry(-0.1847729562531329) q[3];
rz(-2.0953943648345588) q[3];
ry(0.348836453660317) q[4];
rz(0.9052011014403544) q[4];
ry(1.576023632572912) q[5];
rz(0.6957493007629756) q[5];
ry(2.292543444505884) q[6];
rz(-2.2849675621395678) q[6];
ry(-1.3608168985262967) q[7];
rz(2.3111079412584057) q[7];
ry(-1.7107651751538426) q[8];
rz(2.8839808878821467) q[8];
ry(-0.5899968616282704) q[9];
rz(-1.0195408194770996) q[9];
ry(0.22090494747342238) q[10];
rz(-2.330064042029694) q[10];
ry(-1.3261132717820692) q[11];
rz(-3.082008357636065) q[11];