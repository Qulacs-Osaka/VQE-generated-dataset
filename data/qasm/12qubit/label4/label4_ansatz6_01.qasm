OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.6187762954776267) q[0];
ry(-3.071919845328373) q[1];
cx q[0],q[1];
ry(2.1745266009146587) q[0];
ry(-0.10875870900346474) q[1];
cx q[0],q[1];
ry(2.2037247977017165) q[1];
ry(-1.9492442022930374) q[2];
cx q[1],q[2];
ry(-1.1173235069885143) q[1];
ry(2.6709552080636865) q[2];
cx q[1],q[2];
ry(0.02818347897703394) q[2];
ry(-2.762260036341078) q[3];
cx q[2],q[3];
ry(-1.351227351310991) q[2];
ry(-0.11980988168062297) q[3];
cx q[2],q[3];
ry(-1.3214087271438073) q[3];
ry(-2.117011903456077) q[4];
cx q[3],q[4];
ry(1.567881314147482) q[3];
ry(-1.701746461008409) q[4];
cx q[3],q[4];
ry(0.6655371664672609) q[4];
ry(-2.6811010902452956) q[5];
cx q[4],q[5];
ry(-1.6356433138207302) q[4];
ry(-0.00913530112253716) q[5];
cx q[4],q[5];
ry(-0.2040575593608862) q[5];
ry(-0.15420764925217828) q[6];
cx q[5],q[6];
ry(0.5563732319521462) q[5];
ry(2.8589474392769016) q[6];
cx q[5],q[6];
ry(1.8040574310665587) q[6];
ry(-0.004989100787933466) q[7];
cx q[6],q[7];
ry(1.5784219890685298) q[6];
ry(1.57040342305034) q[7];
cx q[6],q[7];
ry(-1.4777938777795159) q[7];
ry(0.1286949422748478) q[8];
cx q[7],q[8];
ry(-0.6324509626632054) q[7];
ry(2.9878319288356923) q[8];
cx q[7],q[8];
ry(-0.5429066937044649) q[8];
ry(-1.6222495337826865) q[9];
cx q[8],q[9];
ry(-1.3999276589361598) q[8];
ry(3.076174543402882) q[9];
cx q[8],q[9];
ry(3.039366125182339) q[9];
ry(-1.1550769705437833) q[10];
cx q[9],q[10];
ry(0.15040109057195253) q[9];
ry(-1.695998734234002) q[10];
cx q[9],q[10];
ry(2.0778999788801613) q[10];
ry(2.4779167826948636) q[11];
cx q[10],q[11];
ry(1.8890157396405192) q[10];
ry(2.566719862422995) q[11];
cx q[10],q[11];
ry(1.020429857528721) q[0];
ry(0.5803225545187072) q[1];
cx q[0],q[1];
ry(3.070139409865132) q[0];
ry(1.6433205874400807) q[1];
cx q[0],q[1];
ry(-2.522723819179736) q[1];
ry(2.915796001967761) q[2];
cx q[1],q[2];
ry(1.5610815062106829) q[1];
ry(-2.393354432837312) q[2];
cx q[1],q[2];
ry(1.5675622867457184) q[2];
ry(1.6721713258581001) q[3];
cx q[2],q[3];
ry(1.5809243855744546) q[2];
ry(-2.4600484517873027) q[3];
cx q[2],q[3];
ry(0.2770929428678155) q[3];
ry(-2.255130738120677) q[4];
cx q[3],q[4];
ry(1.5615750272924966) q[3];
ry(-1.408281364825657) q[4];
cx q[3],q[4];
ry(-0.043190268482891196) q[4];
ry(1.4188411881079368) q[5];
cx q[4],q[5];
ry(1.5609616041472458) q[4];
ry(1.6335179559037334) q[5];
cx q[4],q[5];
ry(2.8796779758740247) q[5];
ry(1.5672806949716167) q[6];
cx q[5],q[6];
ry(1.5577848118487037) q[5];
ry(-2.007602269116348) q[6];
cx q[5],q[6];
ry(1.9145645286562691) q[6];
ry(-0.5411052467642524) q[7];
cx q[6],q[7];
ry(3.1303656795096217) q[6];
ry(1.5692698612088316) q[7];
cx q[6],q[7];
ry(1.1282526749179782) q[7];
ry(-0.1594554441599359) q[8];
cx q[7],q[8];
ry(0.0069352066861561494) q[7];
ry(-3.141222388454406) q[8];
cx q[7],q[8];
ry(-1.715238377015446) q[8];
ry(1.2540898226146897) q[9];
cx q[8],q[9];
ry(0.003659805406641733) q[8];
ry(1.5650040973244899) q[9];
cx q[8],q[9];
ry(-2.743376672082075) q[9];
ry(6.716176011511976e-05) q[10];
cx q[9],q[10];
ry(-1.547557383786721) q[9];
ry(-1.5761708784666306) q[10];
cx q[9],q[10];
ry(-2.4617474883337045) q[10];
ry(-2.678772918418025) q[11];
cx q[10],q[11];
ry(0.9634121577834991) q[10];
ry(0.18379586665380196) q[11];
cx q[10],q[11];
ry(1.2870788906441146) q[0];
ry(2.2576456573279096) q[1];
cx q[0],q[1];
ry(-1.6148773625967614) q[0];
ry(2.8665258353488867) q[1];
cx q[0],q[1];
ry(0.7350689307000157) q[1];
ry(-1.7055790468365322) q[2];
cx q[1],q[2];
ry(-1.4086441856181267) q[1];
ry(-2.842701544351188) q[2];
cx q[1],q[2];
ry(-1.5708021581722664) q[2];
ry(0.6300863651324224) q[3];
cx q[2],q[3];
ry(3.1295202768678183) q[2];
ry(-1.218306631086563) q[3];
cx q[2],q[3];
ry(-2.6259614187865252) q[3];
ry(3.1357076744581205) q[4];
cx q[3],q[4];
ry(1.526850144498372) q[3];
ry(-1.082824856734869) q[4];
cx q[3],q[4];
ry(-3.071047576832266) q[4];
ry(1.566859043796164) q[5];
cx q[4],q[5];
ry(-0.8299177444399164) q[4];
ry(-0.5866723272886718) q[5];
cx q[4],q[5];
ry(-1.5669370300857315) q[5];
ry(0.053504020483763244) q[6];
cx q[5],q[6];
ry(0.20065231140492568) q[5];
ry(1.5529019868981444) q[6];
cx q[5],q[6];
ry(3.0893317114225227) q[6];
ry(0.035411237333343615) q[7];
cx q[6],q[7];
ry(1.017772974164897) q[6];
ry(0.4672772587815142) q[7];
cx q[6],q[7];
ry(1.5828181244163657) q[7];
ry(1.8514776843781806) q[8];
cx q[7],q[8];
ry(-3.138530929350953) q[7];
ry(-2.666656535085957) q[8];
cx q[7],q[8];
ry(-1.2817167127213827) q[8];
ry(-1.4029332070804303) q[9];
cx q[8],q[9];
ry(-0.5385668182754253) q[8];
ry(1.5296345797979418) q[9];
cx q[8],q[9];
ry(-1.5413857669751252) q[9];
ry(-0.0788072983521424) q[10];
cx q[9],q[10];
ry(-1.5823726895549823) q[9];
ry(1.5683335868206327) q[10];
cx q[9],q[10];
ry(1.5695172960043111) q[10];
ry(3.084383108972704) q[11];
cx q[10],q[11];
ry(-1.5719890279131734) q[10];
ry(-1.5715157735302743) q[11];
cx q[10],q[11];
ry(-2.085375378334934) q[0];
ry(0.5931677849589514) q[1];
cx q[0],q[1];
ry(3.058531050133712) q[0];
ry(1.145655620401702) q[1];
cx q[0],q[1];
ry(-2.3373352361305533) q[1];
ry(-0.2451119152964897) q[2];
cx q[1],q[2];
ry(3.1384255596662247) q[1];
ry(-0.6499560845221435) q[2];
cx q[1],q[2];
ry(2.8950778885335953) q[2];
ry(-1.5694329183849156) q[3];
cx q[2],q[3];
ry(1.5136990777227624) q[2];
ry(-1.5945014441938865) q[3];
cx q[2],q[3];
ry(-1.5713006473834445) q[3];
ry(-2.277699009184924) q[4];
cx q[3],q[4];
ry(0.004367079366395465) q[3];
ry(2.057103361200249) q[4];
cx q[3],q[4];
ry(-0.8616297078324385) q[4];
ry(-1.570333180446737) q[5];
cx q[4],q[5];
ry(-0.8402463482789129) q[4];
ry(1.5097746227444668) q[5];
cx q[4],q[5];
ry(1.5744632898518869) q[5];
ry(-1.6486876095588907) q[6];
cx q[5],q[6];
ry(0.009322750224031872) q[5];
ry(-2.882859473774635) q[6];
cx q[5],q[6];
ry(-1.4935192600642884) q[6];
ry(1.5706537387203525) q[7];
cx q[6],q[7];
ry(-2.588980945409861) q[6];
ry(-1.6136953343582634) q[7];
cx q[6],q[7];
ry(1.5707751863127266) q[7];
ry(-1.568441618773225) q[8];
cx q[7],q[8];
ry(1.5816121805156191) q[7];
ry(-0.45812778241730795) q[8];
cx q[7],q[8];
ry(1.5706964539059152) q[8];
ry(-1.5707428407958717) q[9];
cx q[8],q[9];
ry(-1.572021471535912) q[8];
ry(1.5643157904763858) q[9];
cx q[8],q[9];
ry(1.5713444529165272) q[9];
ry(1.7330064137083587) q[10];
cx q[9],q[10];
ry(1.5707619487160218) q[9];
ry(-1.5717595128064907) q[10];
cx q[9],q[10];
ry(1.5707539516223015) q[10];
ry(2.0990808230932427) q[11];
cx q[10],q[11];
ry(1.575384565022308) q[10];
ry(-1.618672686107038) q[11];
cx q[10],q[11];
ry(2.9830029432244594) q[0];
ry(2.251989479241506) q[1];
ry(-1.5730202906684463) q[2];
ry(1.570506993107597) q[3];
ry(1.569671245748141) q[4];
ry(1.5751263313658572) q[5];
ry(1.570681156726451) q[6];
ry(-1.5710276255180577) q[7];
ry(1.5709166042340694) q[8];
ry(-1.5708918121415893) q[9];
ry(1.570800156633554) q[10];
ry(1.5761214589414687) q[11];