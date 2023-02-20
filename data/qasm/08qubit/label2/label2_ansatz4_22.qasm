OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.3472198057272893) q[0];
rz(2.801976298425207) q[0];
ry(-2.7158658019154567) q[1];
rz(0.946970138837445) q[1];
ry(-0.0005296426338279389) q[2];
rz(2.5888186095828702) q[2];
ry(0.0008615589521179245) q[3];
rz(-0.5647493899633369) q[3];
ry(-1.5197198872190105) q[4];
rz(1.5264336230465734) q[4];
ry(2.7899508009263103) q[5];
rz(-0.5985754745606392) q[5];
ry(0.0003492219887949685) q[6];
rz(-0.8216476559337671) q[6];
ry(0.0003717993821915294) q[7];
rz(0.1875769060738252) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.1394307832304453) q[0];
rz(0.29714263064305424) q[0];
ry(3.0125504653671573) q[1];
rz(-2.0306121984451746) q[1];
ry(-2.900325114447999) q[2];
rz(-2.398752607729492) q[2];
ry(-0.19613208941861426) q[3];
rz(1.3159977700619319) q[3];
ry(0.8989418733217206) q[4];
rz(-3.073401641774026) q[4];
ry(-1.556444546991072) q[5];
rz(-1.5600461201463212) q[5];
ry(-1.5664000909501656) q[6];
rz(-0.9589826142963208) q[6];
ry(-1.575370274325012) q[7];
rz(-1.374315773555268) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.5910139653626911) q[0];
rz(-2.5290184805736913) q[0];
ry(-2.988410580695408) q[1];
rz(0.018003038040947498) q[1];
ry(-1.6219609299759472) q[2];
rz(-0.8359005093675007) q[2];
ry(-1.516443900753948) q[3];
rz(-0.7304477987247524) q[3];
ry(-1.572073086008466) q[4];
rz(-1.0467794335096465) q[4];
ry(1.5709178525460912) q[5];
rz(2.095994234476934) q[5];
ry(3.139587014303496) q[6];
rz(-0.8410612928167682) q[6];
ry(3.1370425225502974) q[7];
rz(1.541552211887438) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.5317796839857656) q[0];
rz(-1.076255187473409) q[0];
ry(0.06418287580786419) q[1];
rz(0.8682012661579828) q[1];
ry(1.129551180300215) q[2];
rz(-1.5706484474842364) q[2];
ry(-1.1875388244281027) q[3];
rz(-0.32332233323122433) q[3];
ry(1.5704677058556535) q[4];
rz(-3.1413740013306586) q[4];
ry(-1.5713829519879519) q[5];
rz(-3.1389759214978463) q[5];
ry(-1.557714633395367) q[6];
rz(2.0814617540186884) q[6];
ry(-2.1838095269685303) q[7];
rz(-2.6047296950178174) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.3216562688696296) q[0];
rz(-2.513297285376787) q[0];
ry(0.16922244833121242) q[1];
rz(0.122348110075845) q[1];
ry(-3.0018667108979677) q[2];
rz(-0.47233649973996233) q[2];
ry(0.008305131611051243) q[3];
rz(2.490250340378641) q[3];
ry(1.1084040067338838) q[4];
rz(0.08917148993258836) q[4];
ry(-1.1083506759810993) q[5];
rz(0.25081606077376845) q[5];
ry(-1.4843771682664668) q[6];
rz(-0.6071040709218511) q[6];
ry(1.9419503156031466) q[7];
rz(1.564591260655722) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.4645524764997546) q[0];
rz(-3.003428786923678) q[0];
ry(0.8687648048272321) q[1];
rz(-0.9479361469478373) q[1];
ry(-2.1816363224294437) q[2];
rz(2.9619643866954912) q[2];
ry(-0.9855946686476614) q[3];
rz(0.34371473319026086) q[3];
ry(3.125980358702917) q[4];
rz(1.4225417096403754) q[4];
ry(-0.026946989049650285) q[5];
rz(-2.0811689461444107) q[5];
ry(2.2861810885000695) q[6];
rz(2.5319005812358646) q[6];
ry(-1.6312234715543816) q[7];
rz(-0.6612341601193901) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.325690799837897) q[0];
rz(1.9100770469341524) q[0];
ry(-1.7515914117948104) q[1];
rz(-1.2843605778212295) q[1];
ry(1.9890544811441746) q[2];
rz(0.7905279947383965) q[2];
ry(1.8070896469754671) q[3];
rz(0.689724345934504) q[3];
ry(3.141317463665189) q[4];
rz(-0.19691074424257685) q[4];
ry(3.1399872099626824) q[5];
rz(-0.40198554959943755) q[5];
ry(2.8616827022771605) q[6];
rz(2.7140192366569855) q[6];
ry(2.36936108807397) q[7];
rz(0.4416851223136697) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.2974855265697975) q[0];
rz(0.17572665230293885) q[0];
ry(-0.9155451059405194) q[1];
rz(-0.5635946314553445) q[1];
ry(-0.04005628888915336) q[2];
rz(-1.3558935934821208) q[2];
ry(-0.035649572013298325) q[3];
rz(2.447017207236426) q[3];
ry(-2.1644244976195184) q[4];
rz(-1.6176153016973513) q[4];
ry(0.047952656308921246) q[5];
rz(-1.2431438495058622) q[5];
ry(-1.189874617623077) q[6];
rz(3.0820280057926825) q[6];
ry(1.0565523219065573) q[7];
rz(1.9753414109199192) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.17340766169154515) q[0];
rz(-2.3070374905871627) q[0];
ry(1.7931362006687088) q[1];
rz(1.2320910733818948) q[1];
ry(-2.0418164532416467) q[2];
rz(1.770985426192012) q[2];
ry(3.0047610591219907) q[3];
rz(-2.017415065803253) q[3];
ry(-1.2401716441065902) q[4];
rz(-1.167747105596334) q[4];
ry(2.7535167820744504) q[5];
rz(0.6045433231360349) q[5];
ry(1.8013644181885926) q[6];
rz(2.3504964324088533) q[6];
ry(1.097876283973645) q[7];
rz(-1.9346303941344845) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.3197606408332883) q[0];
rz(0.9954793066715198) q[0];
ry(-2.6128422066381267) q[1];
rz(1.810218189270584) q[1];
ry(-0.035743411938688116) q[2];
rz(-2.0568831094349864) q[2];
ry(1.0976372355860382) q[3];
rz(-0.8444866143253114) q[3];
ry(3.1398217405941082) q[4];
rz(-0.9529026172951536) q[4];
ry(-0.0024989736820328896) q[5];
rz(-0.20909121955585358) q[5];
ry(0.0556256228481851) q[6];
rz(-0.23000962442774672) q[6];
ry(0.05286361021371972) q[7];
rz(-0.8862832111026098) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.1209830279016004) q[0];
rz(-1.9788326947095936) q[0];
ry(-0.4981608418573853) q[1];
rz(-1.5966323768154125) q[1];
ry(1.6080601976301139) q[2];
rz(1.6021442918500446) q[2];
ry(-1.1266288005690424) q[3];
rz(2.8835183124948185) q[3];
ry(0.6457649864785204) q[4];
rz(-1.0025337674139312) q[4];
ry(0.6363269841353034) q[5];
rz(0.20132240925038403) q[5];
ry(1.9437395167597322) q[6];
rz(2.068909979810368) q[6];
ry(-2.218934773106416) q[7];
rz(0.4021802568636449) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.33476871508753797) q[0];
rz(-0.792051594708857) q[0];
ry(-1.6962197782183666) q[1];
rz(0.8957856308478344) q[1];
ry(-1.0542582310099733) q[2];
rz(1.2017644627504982) q[2];
ry(2.2455736591551836) q[3];
rz(2.71102980579028) q[3];
ry(-2.95209607377759) q[4];
rz(0.13262829369727314) q[4];
ry(0.8433754677781617) q[5];
rz(-1.2071290535222374) q[5];
ry(0.04009095192284466) q[6];
rz(-0.38817560293436587) q[6];
ry(-0.08286020017239998) q[7];
rz(-2.9979238751821904) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.0530574418513785) q[0];
rz(-2.1239269361532513) q[0];
ry(2.3151314273610546) q[1];
rz(-3.0599960906150625) q[1];
ry(0.005079943511659982) q[2];
rz(-1.0472899048092423) q[2];
ry(3.081609865297243) q[3];
rz(1.6168419972327186) q[3];
ry(-1.4311882027724305) q[4];
rz(-2.0161425308201837) q[4];
ry(-1.737124017829574) q[5];
rz(0.46639668020790204) q[5];
ry(-3.126499274611968) q[6];
rz(-1.2055712065835975) q[6];
ry(-0.014399729078894459) q[7];
rz(0.09074947698973373) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.29283033365951094) q[0];
rz(2.371848898857839) q[0];
ry(-0.4599758146888968) q[1];
rz(-0.26497112017470226) q[1];
ry(1.6033317066619117) q[2];
rz(1.7465439515006356) q[2];
ry(-0.68954074639592) q[3];
rz(2.2030939680334436) q[3];
ry(2.0607019600173064) q[4];
rz(-0.22740350059553738) q[4];
ry(-0.4367548831135247) q[5];
rz(-2.9135845869313757) q[5];
ry(-2.4589679727146496) q[6];
rz(0.8304409966882211) q[6];
ry(0.6726248653218613) q[7];
rz(-2.734209367613294) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.7063769604386403) q[0];
rz(-2.0770777081978116) q[0];
ry(0.9381600172482392) q[1];
rz(-2.8661488685789496) q[1];
ry(0.12036730833605258) q[2];
rz(-1.1001408082846567) q[2];
ry(-1.4084251675616635) q[3];
rz(-1.3399196990711966) q[3];
ry(3.141293329603982) q[4];
rz(1.9117980128805017) q[4];
ry(-3.139808571335734) q[5];
rz(0.49120542130494105) q[5];
ry(-0.006392402110517281) q[6];
rz(-1.656365472109631) q[6];
ry(-3.1081775556781333) q[7];
rz(1.1893514083621692) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.7422968897912927) q[0];
rz(-1.0330040825441342) q[0];
ry(-1.8338642162292276) q[1];
rz(-0.6435520400384277) q[1];
ry(2.3133188661745456) q[2];
rz(1.1256110785743412) q[2];
ry(0.23816723729045994) q[3];
rz(2.6502373392589944) q[3];
ry(-2.2159438745322912) q[4];
rz(1.6223613611673593) q[4];
ry(-2.2610722140029686) q[5];
rz(1.6629886599641839) q[5];
ry(-2.7926883607575133) q[6];
rz(0.9799354911840172) q[6];
ry(0.7514697876782326) q[7];
rz(-0.8870527959764507) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.9689392910923371) q[0];
rz(0.21765526164010296) q[0];
ry(-2.4124293497837432) q[1];
rz(1.0936172019857422) q[1];
ry(2.8824673543503807) q[2];
rz(-2.3984070911735427) q[2];
ry(2.5653697330543817) q[3];
rz(-2.517864217668753) q[3];
ry(2.460894914724376) q[4];
rz(-2.4849141308905094) q[4];
ry(-0.7025786342263067) q[5];
rz(-1.2129735264144987) q[5];
ry(1.4073090770312646) q[6];
rz(-0.09041673737542945) q[6];
ry(-1.762221548803697) q[7];
rz(0.0026813216917725663) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.8999674989711313) q[0];
rz(-0.3871866390328842) q[0];
ry(2.866961617804718) q[1];
rz(2.332359162802552) q[1];
ry(-1.8692097711929998) q[2];
rz(1.1881305099809707) q[2];
ry(2.5087776903337575) q[3];
rz(-1.18133388326096) q[3];
ry(0.000307395233916985) q[4];
rz(-1.6058096122506944) q[4];
ry(0.0057493235276302465) q[5];
rz(0.193385551666454) q[5];
ry(2.238561291970809) q[6];
rz(0.9615362884693726) q[6];
ry(2.2327689402647626) q[7];
rz(-2.8148202445572763) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.8801878930464646) q[0];
rz(2.6908178306079225) q[0];
ry(0.09987879639646025) q[1];
rz(-2.0910135894045485) q[1];
ry(2.450077812447239) q[2];
rz(1.1308206970399608) q[2];
ry(-1.9589690712565346) q[3];
rz(1.9722216699227806) q[3];
ry(-2.144762419627978) q[4];
rz(-0.6589112757116806) q[4];
ry(-0.9982640910271171) q[5];
rz(-2.1666628459678456) q[5];
ry(-0.4098478545335683) q[6];
rz(1.7197461348134366) q[6];
ry(0.21745081869504346) q[7];
rz(2.2137085576416853) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.6259271597747258) q[0];
rz(-0.3122003368510654) q[0];
ry(-0.7743215008127473) q[1];
rz(-2.255614240466235) q[1];
ry(2.4462166194056283) q[2];
rz(-0.8182863834636366) q[2];
ry(-2.072263314336814) q[3];
rz(2.743003411872404) q[3];
ry(0.40031138235550046) q[4];
rz(1.29979511018721) q[4];
ry(-2.0039801052750947) q[5];
rz(1.61877103637969) q[5];
ry(0.010268430877078849) q[6];
rz(1.8163085538834567) q[6];
ry(-0.0025421818926698596) q[7];
rz(2.000548234291616) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.072671782882063) q[0];
rz(2.029429830200192) q[0];
ry(2.078489634108351) q[1];
rz(2.6230281466898098) q[1];
ry(-3.1256712356205645) q[2];
rz(-0.34723449557373853) q[2];
ry(-0.0051152280708644205) q[3];
rz(-2.841990279811148) q[3];
ry(-2.1853329707641675) q[4];
rz(3.125859973291205) q[4];
ry(2.196189974115968) q[5];
rz(-1.6540023275724254) q[5];
ry(1.585918651017466) q[6];
rz(0.4214515357987087) q[6];
ry(1.5572718957855862) q[7];
rz(-1.2524087847661374) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.5733598807451212) q[0];
rz(0.5334420829110709) q[0];
ry(-2.9669993842217486) q[1];
rz(-2.402547456403868) q[1];
ry(-2.148711680476884) q[2];
rz(1.3232933401432616) q[2];
ry(0.15946788998116104) q[3];
rz(1.7129946782869059) q[3];
ry(1.557592677655975) q[4];
rz(-2.4617481987736163) q[4];
ry(-1.0187094329803459) q[5];
rz(-0.42125768013128967) q[5];
ry(1.1763020853544663) q[6];
rz(0.6958402310419158) q[6];
ry(0.8307491170240405) q[7];
rz(-0.21122089160456528) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.2629391131044563) q[0];
rz(-2.671737924019971) q[0];
ry(1.0232746852704282) q[1];
rz(0.4818724928956329) q[1];
ry(3.108990995630534) q[2];
rz(3.0528159676044893) q[2];
ry(-3.1253928464017764) q[3];
rz(-2.058318125168309) q[3];
ry(-3.1397887949152996) q[4];
rz(1.745943650054964) q[4];
ry(-3.139197548378371) q[5];
rz(2.215758659555343) q[5];
ry(1.5534675810148464) q[6];
rz(0.7640914899426078) q[6];
ry(-1.5531963345886648) q[7];
rz(-2.6022313073297147) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.6081498494770035) q[0];
rz(-2.1370579138236194) q[0];
ry(1.6792400571027282) q[1];
rz(0.1510307802043635) q[1];
ry(-1.2424976561223868) q[2];
rz(-3.1301652224061574) q[2];
ry(0.4237298835143273) q[3];
rz(1.8300550546658894) q[3];
ry(-0.9459929291213482) q[4];
rz(1.3843591033545852) q[4];
ry(-0.13000088880695773) q[5];
rz(1.8611065707788566) q[5];
ry(2.521065462366832) q[6];
rz(-2.7912733381856185) q[6];
ry(0.10864903835551143) q[7];
rz(2.415962663907599) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.4251118571307675) q[0];
rz(-2.6853379228608603) q[0];
ry(-2.9332973932962303) q[1];
rz(-2.7551089091390817) q[1];
ry(-0.8310444745477312) q[2];
rz(-1.1918569062496438) q[2];
ry(0.8500206489240485) q[3];
rz(1.9933709997735807) q[3];
ry(3.1404635224850197) q[4];
rz(0.31685755080712286) q[4];
ry(3.1412309202979434) q[5];
rz(1.0173307432907315) q[5];
ry(3.1396329695530762) q[6];
rz(0.277435690558393) q[6];
ry(0.002829949506205409) q[7];
rz(-2.053313595467592) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.0997687392565867) q[0];
rz(1.6559933643523328) q[0];
ry(3.034048405322152) q[1];
rz(2.5495135605582475) q[1];
ry(1.0417993128729224) q[2];
rz(-2.314149853112096) q[2];
ry(2.216055181519695) q[3];
rz(-1.8697176360388077) q[3];
ry(2.629744852287276) q[4];
rz(-2.4795685523954503) q[4];
ry(1.2469081824533594) q[5];
rz(-3.099942797892279) q[5];
ry(2.10692777910128) q[6];
rz(2.3049165573227723) q[6];
ry(0.7168696993486687) q[7];
rz(-2.5720177832224014) q[7];