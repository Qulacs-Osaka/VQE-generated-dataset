OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-3.1314577572716997) q[0];
rz(-0.08810806382249026) q[0];
ry(1.5707590940573923) q[1];
rz(-0.00563240503572934) q[1];
ry(-0.24183151221004984) q[2];
rz(-2.543098925596451) q[2];
ry(-1.5722615263668906) q[3];
rz(-2.1522274631656098) q[3];
ry(-3.133417335962244) q[4];
rz(0.1076703556560957) q[4];
ry(-3.054729427178382) q[5];
rz(1.706175601261367) q[5];
ry(0.031245364061471874) q[6];
rz(1.49666236983336) q[6];
ry(3.133217620265693) q[7];
rz(2.279605961358203) q[7];
ry(-3.1414835428132513) q[8];
rz(-2.837651864952754) q[8];
ry(-3.140593586975955) q[9];
rz(3.0632339915820843) q[9];
ry(0.0005937904764383717) q[10];
rz(-1.4204394478060078) q[10];
ry(-3.1412628821725592) q[11];
rz(-2.5296716480886716) q[11];
ry(0.00048053750921983607) q[12];
rz(-1.14753307922871) q[12];
ry(-3.1398138949812986) q[13];
rz(-0.838865751314734) q[13];
ry(0.0011636169235664795) q[14];
rz(-2.8199445744374483) q[14];
ry(3.140822273993474) q[15];
rz(-2.503456366302961) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.140911387773917) q[0];
rz(-0.6648420128353869) q[0];
ry(-1.573102212687004) q[1];
rz(-1.568559679013874) q[1];
ry(0.006063164780868213) q[2];
rz(1.365440936723273) q[2];
ry(3.1348052652603244) q[3];
rz(2.55908141641964) q[3];
ry(-0.0005392546159344036) q[4];
rz(-1.4203263643240618) q[4];
ry(0.3193778538702414) q[5];
rz(-3.137806545997332) q[5];
ry(0.791168754941411) q[6];
rz(-0.00375950861669768) q[6];
ry(-0.3763936212283916) q[7];
rz(0.030234038460697835) q[7];
ry(-2.9958894016321658) q[8];
rz(3.0038863870305548) q[8];
ry(-3.131431926503246) q[9];
rz(1.8120908328385648) q[9];
ry(3.1307501724868327) q[10];
rz(-2.3010882012430494) q[10];
ry(-3.139576198955369) q[11];
rz(1.211426877174661) q[11];
ry(3.1392251521497645) q[12];
rz(-1.336706437278562) q[12];
ry(-3.135553914645699) q[13];
rz(1.1728853762704388) q[13];
ry(-3.1397616600986966) q[14];
rz(2.780559873696616) q[14];
ry(0.0006644338858057902) q[15];
rz(-1.4222300176259122) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.1414257388339784) q[0];
rz(3.1139212094206554) q[0];
ry(-1.0314619056590475) q[1];
rz(-3.0084281329817237) q[1];
ry(0.0019029843689724046) q[2];
rz(-2.0841382275603717) q[2];
ry(-1.0727053892106202) q[3];
rz(1.8913893107881796) q[3];
ry(-3.141415916113994) q[4];
rz(1.310204946490873) q[4];
ry(0.21614464664729136) q[5];
rz(0.13539590295674309) q[5];
ry(-2.4116429173588014) q[6];
rz(-2.9492258866636236) q[6];
ry(0.2620763481290162) q[7];
rz(3.130653068539738) q[7];
ry(-0.09918638905914179) q[8];
rz(0.05578819461447296) q[8];
ry(-0.016714856354571594) q[9];
rz(0.08072577017078256) q[9];
ry(-0.08960049135592385) q[10];
rz(-1.7037085330624968) q[10];
ry(-0.22908921257555992) q[11];
rz(2.1521398233783136) q[11];
ry(-2.714065191904033) q[12];
rz(2.839430040647419) q[12];
ry(-2.5475092997119435) q[13];
rz(0.04859926534503535) q[13];
ry(0.4051471128610135) q[14];
rz(-3.0864405371893797) q[14];
ry(-0.19030678567012774) q[15];
rz(0.08187424005496732) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-0.00048440936332784425) q[0];
rz(-0.001599322978848683) q[0];
ry(3.1397129052541852) q[1];
rz(-0.425240018397516) q[1];
ry(-3.1396515161878695) q[2];
rz(-0.5916347236837254) q[2];
ry(3.1394676109048922) q[3];
rz(2.1628527794729324) q[3];
ry(0.00033524657909556055) q[4];
rz(-2.2134595010316174) q[4];
ry(-3.1099322004347605) q[5];
rz(-0.15676945340693615) q[5];
ry(3.0722408478268157) q[6];
rz(-0.0898562859435561) q[6];
ry(2.952695436080335) q[7];
rz(0.10715610455436408) q[7];
ry(0.38723532083845935) q[8];
rz(0.03692346902354579) q[8];
ry(-2.5230278110431255) q[9];
rz(3.1249476478756684) q[9];
ry(-2.692723231185749) q[10];
rz(-0.1178629633051302) q[10];
ry(-0.2886134074128503) q[11];
rz(-0.5305187033897157) q[11];
ry(-2.7964012728501717) q[12];
rz(-2.2800878516224294) q[12];
ry(-2.637841386284456) q[13];
rz(0.9074683369094587) q[13];
ry(-0.3103478378896708) q[14];
rz(-0.29109392985272026) q[14];
ry(-0.13068491290013987) q[15];
rz(-3.0435072737155493) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-0.0015183620129350928) q[0];
rz(-0.6047289484674483) q[0];
ry(-2.7463790720995243) q[1];
rz(2.6532824605836125) q[1];
ry(0.03745417771856837) q[2];
rz(0.12239404489652285) q[2];
ry(3.0273565201092145) q[3];
rz(2.360154062208644) q[3];
ry(-1.5770697846522015) q[4];
rz(-3.1408315802011475) q[4];
ry(0.23932527178453) q[5];
rz(0.3605863827252103) q[5];
ry(-0.04054351874851336) q[6];
rz(-1.9916217860165604) q[6];
ry(-3.0386693653429013) q[7];
rz(-1.5931819081191019) q[7];
ry(2.8700285755876167) q[8];
rz(-1.9006357378467922) q[8];
ry(2.62624925702894) q[9];
rz(1.502742229939189) q[9];
ry(2.80996129248341) q[10];
rz(-1.0280030422915187) q[10];
ry(0.15897885853995408) q[11];
rz(-2.0445616558126223) q[11];
ry(-0.05510473194100029) q[12];
rz(2.454311741261405) q[12];
ry(-0.03525764003527522) q[13];
rz(-1.2249409223978407) q[13];
ry(3.1185008193775445) q[14];
rz(1.9835794879600197) q[14];
ry(0.007104816591418405) q[15];
rz(-0.25434548651499966) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(0.0019527974642132558) q[0];
rz(-2.872964988275374) q[0];
ry(1.555445078593383) q[1];
rz(-1.371028885745849) q[1];
ry(3.141346816825633) q[2];
rz(-0.7064636575911956) q[2];
ry(-0.026443846543019234) q[3];
rz(1.1038767497441477) q[3];
ry(1.5783858408541418) q[4];
rz(1.4987791826148436) q[4];
ry(-0.0015148663684891256) q[5];
rz(1.6322441069965636) q[5];
ry(3.141571460905598) q[6];
rz(2.409625808416781) q[6];
ry(-3.1415776152588992) q[7];
rz(-0.9293157319182401) q[7];
ry(-3.141500445809212) q[8];
rz(0.8384466618161968) q[8];
ry(-7.946537607228521e-05) q[9];
rz(0.29651995265026615) q[9];
ry(-3.1415378542228134) q[10];
rz(3.0239388790906148) q[10];
ry(-8.476213168134726e-05) q[11];
rz(-1.1232686863633186) q[11];
ry(7.213533141037232e-05) q[12];
rz(-0.5059528422378357) q[12];
ry(-3.1414717129024647) q[13];
rz(0.4298694272055812) q[13];
ry(3.141477730899361) q[14];
rz(-1.2364802630044989) q[14];
ry(3.141530857623606) q[15];
rz(-0.5589466993273975) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-0.2514366094003933) q[0];
rz(0.09157909447992148) q[0];
ry(-1.0604991942424447) q[1];
rz(2.99512880542931) q[1];
ry(-2.882438860132681) q[2];
rz(-1.4701740118303066) q[2];
ry(-1.5357910590324295) q[3];
rz(-3.0748601069781674) q[3];
ry(-0.2050969999264918) q[4];
rz(-2.3014749175986298) q[4];
ry(0.0128719534691486) q[5];
rz(-1.4865891576246624) q[5];
ry(-3.1364761778065313) q[6];
rz(-1.0613583728468798) q[6];
ry(0.001967576356173062) q[7];
rz(1.8179330375431242) q[7];
ry(3.1415500374650067) q[8];
rz(2.410404699260737) q[8];
ry(-0.0004281450593930071) q[9];
rz(-0.388352793891138) q[9];
ry(-3.1415203158454847) q[10];
rz(-2.181756439375288) q[10];
ry(-3.141248957322908) q[11];
rz(0.03668299373302173) q[11];
ry(3.141482628683891) q[12];
rz(0.7987953838035207) q[12];
ry(-3.1414111991798404) q[13];
rz(2.347553788485634) q[13];
ry(-0.00012551675675592127) q[14];
rz(-2.381020178382386) q[14];
ry(-3.141282977274832) q[15];
rz(1.3010633214801992) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(2.9148424822978707) q[0];
rz(-2.3726442460431336) q[0];
ry(2.6013773487366256) q[1];
rz(-0.25447659195795946) q[1];
ry(-2.7028467801724) q[2];
rz(-2.4505146057770726) q[2];
ry(-0.8592444824888618) q[3];
rz(-3.0922630535211404) q[3];
ry(-0.2858698384596951) q[4];
rz(-2.5443019591482776) q[4];
ry(-1.8554104441447477) q[5];
rz(0.5397468365106436) q[5];
ry(1.3664139588112247) q[6];
rz(-1.7369434001642459) q[6];
ry(-0.14548261671063498) q[7];
rz(2.916327977264902) q[7];
ry(0.8968539735277631) q[8];
rz(-2.7916468939947126) q[8];
ry(-3.0683307737468812) q[9];
rz(-2.082560523434241) q[9];
ry(0.024210613217114606) q[10];
rz(3.0347614323126253) q[10];
ry(-1.6567863798570484) q[11];
rz(1.1083390636423767) q[11];
ry(-2.529280809719349) q[12];
rz(-1.6114947384162466) q[12];
ry(1.6171828303281963) q[13];
rz(1.6442450319222441) q[13];
ry(0.2091243341602324) q[14];
rz(-3.0061124491441755) q[14];
ry(2.2693249333367436) q[15];
rz(1.6129991264671908) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.141584112775953) q[0];
rz(-0.9108513248070601) q[0];
ry(0.00021754673482465847) q[1];
rz(0.1768227625042584) q[1];
ry(3.1415702918609876) q[2];
rz(-2.591615398175779) q[2];
ry(0.00013869589552975887) q[3];
rz(0.3671367853898315) q[3];
ry(5.44997915612176e-05) q[4];
rz(3.0413331317498766) q[4];
ry(4.727609777077645e-05) q[5];
rz(0.04184782501251227) q[5];
ry(6.091702467592802e-07) q[6];
rz(0.48833688234677464) q[6];
ry(-3.141538463286716) q[7];
rz(-0.8090091670155791) q[7];
ry(3.1415016510923977) q[8];
rz(2.986376536397138) q[8];
ry(-3.1415741336846965) q[9];
rz(-0.3635699987385479) q[9];
ry(5.8351195131378325e-05) q[10];
rz(0.06900409707681902) q[10];
ry(-4.5613032329860914e-05) q[11];
rz(-1.1095222777481277) q[11];
ry(3.1415740375517407) q[12];
rz(-0.7811738044122452) q[12];
ry(3.141520100852821) q[13];
rz(2.003464761687871) q[13];
ry(5.8699211072266166e-05) q[14];
rz(-2.1412495726892624) q[14];
ry(3.141561533897294) q[15];
rz(-1.8576288571727506) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.5465472094373096) q[0];
rz(-0.1880023370036357) q[0];
ry(2.090579118626931) q[1];
rz(1.6903913670271542) q[1];
ry(2.006638929419755) q[2];
rz(-0.0254425976258455) q[2];
ry(0.7567267319397581) q[3];
rz(2.8253073365193404) q[3];
ry(-1.3188483743550465) q[4];
rz(-0.14024092074096028) q[4];
ry(0.33646934424603936) q[5];
rz(-0.5563706414697434) q[5];
ry(0.5806366548145058) q[6];
rz(-1.9498393625811916) q[6];
ry(-1.4492129921421324) q[7];
rz(-3.061113227665182) q[7];
ry(-0.7404005745166442) q[8];
rz(0.38783018097459193) q[8];
ry(-1.5811274442669483) q[9];
rz(-0.07262155141606262) q[9];
ry(1.5944382588534045) q[10];
rz(-0.0008912658426121212) q[10];
ry(0.08581411046532493) q[11];
rz(-3.140015445895364) q[11];
ry(-1.12883412861857) q[12];
rz(-0.4381074222686871) q[12];
ry(0.04925434486352798) q[13];
rz(2.7830848840709015) q[13];
ry(-1.659669390612488) q[14];
rz(-2.9523146362264576) q[14];
ry(-0.7258537101762474) q[15];
rz(-2.8913746229142836) q[15];