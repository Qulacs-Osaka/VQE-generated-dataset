OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.9703144284008531) q[0];
rz(-0.5880149512815445) q[0];
ry(-2.7170557931125563) q[1];
rz(1.1812757436317458) q[1];
ry(-2.955610079976699) q[2];
rz(0.5806285097706678) q[2];
ry(3.0782932430268444) q[3];
rz(-0.9149867598178213) q[3];
ry(1.5662998505220243) q[4];
rz(3.140634978402118) q[4];
ry(1.5693695462483706) q[5];
rz(3.1374358393670057) q[5];
ry(-1.5591895957966293) q[6];
rz(-0.6816144574462211) q[6];
ry(-1.5319036206580996) q[7];
rz(3.113955775458944) q[7];
ry(-0.8447061091344761) q[8];
rz(1.3969080652772359) q[8];
ry(1.4141282387685834) q[9];
rz(3.0632003087537685) q[9];
ry(-0.3338705083537137) q[10];
rz(0.7845155964800385) q[10];
ry(-2.9117913655105307) q[11];
rz(-0.8120251694908474) q[11];
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
ry(3.0981467678984305) q[0];
rz(1.0664467209531932) q[0];
ry(0.10581191379069343) q[1];
rz(1.516692955193079) q[1];
ry(0.031822019139230585) q[2];
rz(1.8489580101438925) q[2];
ry(0.028419234004278857) q[3];
rz(-2.0333065499941423) q[3];
ry(-1.735282119537062) q[4];
rz(2.077585113766655) q[4];
ry(1.7329082856439442) q[5];
rz(-2.0403895721375394) q[5];
ry(3.1408755743422345) q[6];
rz(-2.2337466081616064) q[6];
ry(-0.0007736789032033925) q[7];
rz(1.6191393354141725) q[7];
ry(0.005969675426524917) q[8];
rz(-2.445151892909469) q[8];
ry(-0.010980883270452324) q[9];
rz(2.6241644542623406) q[9];
ry(1.9563535811509425) q[10];
rz(0.22262201415433314) q[10];
ry(1.7056101791989111) q[11];
rz(-0.2077897181617986) q[11];
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
ry(0.8537753082471178) q[0];
rz(-2.8992692284567685) q[0];
ry(-0.5375905618405055) q[1];
rz(-1.428304923604565) q[1];
ry(-2.2477056497627297) q[2];
rz(-0.6838390587869767) q[2];
ry(-0.6861307266574969) q[3];
rz(1.9325943115545154) q[3];
ry(0.8382810724947811) q[4];
rz(-2.4426983984755126) q[4];
ry(-2.4562015987507526) q[5];
rz(1.5738001486568647) q[5];
ry(2.776831233539036) q[6];
rz(3.0488177245594374) q[6];
ry(2.868345338471095) q[7];
rz(1.1252957656044027) q[7];
ry(-1.3436785266358275) q[8];
rz(-1.265387586798509) q[8];
ry(0.04174371078983253) q[9];
rz(2.562812349190765) q[9];
ry(-2.399178476847272) q[10];
rz(-2.7834360195597876) q[10];
ry(0.9588137730951991) q[11];
rz(2.686816868182423) q[11];
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
ry(-0.013653099054587987) q[0];
rz(0.37849073913055875) q[0];
ry(-0.03972853748955529) q[1];
rz(0.5491359101675677) q[1];
ry(0.0014807594567889384) q[2];
rz(0.41612843856574605) q[2];
ry(-0.0013732897181331083) q[3];
rz(-0.5132331563763621) q[3];
ry(0.0010436170546612026) q[4];
rz(1.460015412520252) q[4];
ry(3.1409055521192153) q[5];
rz(0.6216680593770265) q[5];
ry(-0.001460560567922542) q[6];
rz(0.44964400842502467) q[6];
ry(3.1414303589722676) q[7];
rz(-2.5514379789533392) q[7];
ry(3.1314518225091654) q[8];
rz(-0.7469149341421444) q[8];
ry(-3.131602743999925) q[9];
rz(0.32245796166712726) q[9];
ry(-2.709517351708682) q[10];
rz(-1.9792656802971444) q[10];
ry(1.1535467351330002) q[11];
rz(0.4364721645652248) q[11];
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
ry(-2.434727598234482) q[0];
rz(0.5239203711111688) q[0];
ry(-2.9422736701039818) q[1];
rz(-1.3317824294024958) q[1];
ry(-0.5176561887115008) q[2];
rz(-2.741936022561428) q[2];
ry(1.4081085376339966) q[3];
rz(1.1630842983430245) q[3];
ry(-1.7488716244541507) q[4];
rz(-2.3782594151381224) q[4];
ry(0.7908922764798216) q[5];
rz(-1.256909030054915) q[5];
ry(-0.3070456662276527) q[6];
rz(-1.243774671645938) q[6];
ry(2.321682433290748) q[7];
rz(1.6319795533742194) q[7];
ry(-1.5029791704569166) q[8];
rz(1.7329767618709262) q[8];
ry(1.958451072940087) q[9];
rz(-1.383693420943671) q[9];
ry(2.9453775662590127) q[10];
rz(1.6590097538339101) q[10];
ry(3.092181237892882) q[11];
rz(-2.695388832526638) q[11];
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
ry(1.5811758955308568) q[0];
rz(1.6187151552625627) q[0];
ry(-3.08217473101863) q[1];
rz(-0.17131234462988132) q[1];
ry(0.002654591772262904) q[2];
rz(0.9769764074126385) q[2];
ry(0.004130087967444397) q[3];
rz(1.8505964866283173) q[3];
ry(3.14078352096188) q[4];
rz(1.1454601609492425) q[4];
ry(-2.2502491527909282e-05) q[5];
rz(-0.8181665838300302) q[5];
ry(2.565015883814365e-05) q[6];
rz(3.0156438671422663) q[6];
ry(-3.1410990382112733) q[7];
rz(1.3801424628620644) q[7];
ry(0.005028589015960705) q[8];
rz(1.5346465045278816) q[8];
ry(0.004058546309126498) q[9];
rz(-1.5010149411843505) q[9];
ry(1.419408792266263) q[10];
rz(1.0171788959254522) q[10];
ry(1.388315995278634) q[11];
rz(0.7355158101623207) q[11];
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
ry(1.2724555296801083) q[0];
rz(0.9725225409647901) q[0];
ry(-2.3251985873863927) q[1];
rz(1.658485222054858) q[1];
ry(-1.3388002036436477) q[2];
rz(-1.9827045732686464) q[2];
ry(2.7752946548891058) q[3];
rz(1.0630320145957814) q[3];
ry(-1.3411038206605115) q[4];
rz(3.02480234766837) q[4];
ry(-2.0154000778267225) q[5];
rz(-1.3725733845703294) q[5];
ry(-3.001146104117084) q[6];
rz(-3.10391476423706) q[6];
ry(-0.5875047915599705) q[7];
rz(2.8801233056827154) q[7];
ry(0.9122413953927708) q[8];
rz(1.9711584095968313) q[8];
ry(-1.752967322673414) q[9];
rz(1.8703275615002448) q[9];
ry(-1.0525891914506547) q[10];
rz(1.3114301470433443) q[10];
ry(-0.7740831841591724) q[11];
rz(3.134620350154258) q[11];
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
ry(-0.6841313973158057) q[0];
rz(1.933834201295482) q[0];
ry(-1.8412013667766656) q[1];
rz(-2.036385815356593) q[1];
ry(3.0961154059008025) q[2];
rz(-1.1945356473169935) q[2];
ry(0.0018498189488100891) q[3];
rz(0.2467327012964793) q[3];
ry(-3.1253735522590658) q[4];
rz(1.977153045184191) q[4];
ry(3.1389745490634056) q[5];
rz(2.2864047968269947) q[5];
ry(0.020183940476150575) q[6];
rz(-1.0609047002844068) q[6];
ry(-0.02138875271447075) q[7];
rz(1.0006027040569743) q[7];
ry(-0.1599603158591476) q[8];
rz(-2.8032796420521864) q[8];
ry(1.5626120715335636) q[9];
rz(2.0690891066279136) q[9];
ry(1.679809479261536) q[10];
rz(2.1788352970799645) q[10];
ry(0.09303377910080783) q[11];
rz(0.0037846570996284967) q[11];
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
ry(-1.2923493167826416) q[0];
rz(1.6030919179092558) q[0];
ry(2.9301648696536353) q[1];
rz(1.43349689698651) q[1];
ry(-0.6722852061070473) q[2];
rz(-3.0329207047219033) q[2];
ry(-2.471726543803834) q[3];
rz(-2.9650586173962217) q[3];
ry(3.0946868954310456) q[4];
rz(-2.4880217530655573) q[4];
ry(-0.7200321099699737) q[5];
rz(-0.9287364887607922) q[5];
ry(-3.1197054137868463) q[6];
rz(-2.342293229281931) q[6];
ry(3.1214965087711626) q[7];
rz(-0.7965951428640565) q[7];
ry(1.9080434811656675) q[8];
rz(0.6341009769284061) q[8];
ry(-0.4505358044625316) q[9];
rz(-1.6993788416843012) q[9];
ry(0.9196088076421094) q[10];
rz(0.3717586929692809) q[10];
ry(0.6593993888397547) q[11];
rz(-0.5984622609414988) q[11];
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
ry(1.6705071990932598) q[0];
rz(-2.653908020679746) q[0];
ry(0.897432402796162) q[1];
rz(-1.3783442332819034) q[1];
ry(2.8630755291811325) q[2];
rz(2.4277789469692936) q[2];
ry(0.2886582540283982) q[3];
rz(0.5946854679080148) q[3];
ry(-0.0485656822376561) q[4];
rz(1.968437997064985) q[4];
ry(0.04873779841542983) q[5];
rz(0.9748514362130997) q[5];
ry(-0.09056428528938194) q[6];
rz(0.7741307348547557) q[6];
ry(3.0554898850606884) q[7];
rz(-2.3901878360469992) q[7];
ry(-1.9116458958572187) q[8];
rz(3.066599171097106) q[8];
ry(0.5151376120697337) q[9];
rz(2.4781195524791433) q[9];
ry(-3.1227947382941617) q[10];
rz(2.817828618661495) q[10];
ry(-3.0930969320777795) q[11];
rz(1.2505274470559637) q[11];
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
ry(-0.8185891792997791) q[0];
rz(1.3812643918036853) q[0];
ry(0.11918682388376833) q[1];
rz(-0.3685382276407108) q[1];
ry(2.0632402931920852) q[2];
rz(2.4260779396373713) q[2];
ry(1.0571469924056547) q[3];
rz(2.097185537687339) q[3];
ry(0.30258136442920325) q[4];
rz(1.179260573035915) q[4];
ry(-0.2602308796541162) q[5];
rz(-1.269595168811541) q[5];
ry(-1.5508388031828406) q[6];
rz(-1.6699210907767323) q[6];
ry(-1.547612628898886) q[7];
rz(3.0742580172481717) q[7];
ry(-2.482719490280535) q[8];
rz(-1.7806555521353518) q[8];
ry(-2.413033835670892) q[9];
rz(1.8384916196659153) q[9];
ry(1.512444311616976) q[10];
rz(2.9214727412042) q[10];
ry(-2.4081735336555474) q[11];
rz(-1.4371424210000139) q[11];
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
ry(-0.396863179710639) q[0];
rz(-2.4771705264925816) q[0];
ry(-1.3594685921116658) q[1];
rz(-1.3272489527282265) q[1];
ry(-0.002034917156806585) q[2];
rz(-0.4431695926388776) q[2];
ry(0.01529999775470525) q[3];
rz(0.213807845004383) q[3];
ry(-2.440216569697772) q[4];
rz(-1.3159860028116483) q[4];
ry(-0.9198473198964038) q[5];
rz(-0.8682670530509267) q[5];
ry(-3.120620857853027) q[6];
rz(-1.1233525501646504) q[6];
ry(-3.0968864680732837) q[7];
rz(0.4126744569759006) q[7];
ry(1.5545507221149562) q[8];
rz(-0.7695609856573105) q[8];
ry(1.5808380988433184) q[9];
rz(2.6821871942625894) q[9];
ry(-2.0652614630853305) q[10];
rz(1.7506109277888495) q[10];
ry(-2.086568807504812) q[11];
rz(-0.8147488226353133) q[11];
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
ry(2.418859193261127) q[0];
rz(1.188568068233818) q[0];
ry(-2.6743198795379017) q[1];
rz(-1.7963019741597501) q[1];
ry(1.9655696660903326) q[2];
rz(-1.8916155061326771) q[2];
ry(-2.005064162778587) q[3];
rz(-0.14072803094909858) q[3];
ry(-1.1410403989297837) q[4];
rz(0.5352630900794724) q[4];
ry(-1.2904192489614879) q[5];
rz(3.017280550106583) q[5];
ry(-3.1414928698689755) q[6];
rz(0.1984761688767316) q[6];
ry(-0.004521234314288675) q[7];
rz(-1.620051154299058) q[7];
ry(-1.436027884672934) q[8];
rz(1.8346305137747634) q[8];
ry(0.8805400040690827) q[9];
rz(-2.6886806770452685) q[9];
ry(3.1120354818243054) q[10];
rz(-2.3781528545226958) q[10];
ry(0.03503747351068587) q[11];
rz(-0.1289543609067394) q[11];
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
ry(-1.8176928221830142) q[0];
rz(-2.962176524428611) q[0];
ry(-1.8354810089740194) q[1];
rz(0.3448132847529797) q[1];
ry(-3.118995904925749) q[2];
rz(2.5058836658605586) q[2];
ry(0.052171158014665764) q[3];
rz(0.29914301362130225) q[3];
ry(-0.035460280695718716) q[4];
rz(-0.8241606298470461) q[4];
ry(-0.1039047541581466) q[5];
rz(2.894568216461462) q[5];
ry(3.090359542379025) q[6];
rz(1.537969929412286) q[6];
ry(-3.1398120044432365) q[7];
rz(2.2943396482301925) q[7];
ry(1.3829309105586474) q[8];
rz(1.4268471970852534) q[8];
ry(-2.0409204422891376) q[9];
rz(2.688022469116767) q[9];
ry(1.6490320634153839) q[10];
rz(2.9327479273130197) q[10];
ry(2.776216697219735) q[11];
rz(2.9934204837469016) q[11];
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
ry(-3.044181093977013) q[0];
rz(2.1761323257949208) q[0];
ry(3.0593891641352458) q[1];
rz(1.9473848547377464) q[1];
ry(1.4602319271666762) q[2];
rz(1.7710028589949636) q[2];
ry(1.334302277781197) q[3];
rz(-1.3551262482810733) q[3];
ry(1.5706084560536908) q[4];
rz(0.47333977752166856) q[4];
ry(1.4999398590201358) q[5];
rz(-0.768915214997442) q[5];
ry(0.08908485817591405) q[6];
rz(0.753881414708462) q[6];
ry(3.045619996278907) q[7];
rz(2.0307186469391842) q[7];
ry(3.138760536033499) q[8];
rz(-1.1676276720725474) q[8];
ry(0.0032387165132613405) q[9];
rz(2.5266104137312166) q[9];
ry(-1.8902094083973675) q[10];
rz(1.854100489788137) q[10];
ry(0.6402134878768565) q[11];
rz(3.0932737545398457) q[11];
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
ry(0.08219812504446633) q[0];
rz(-1.5300911568661761) q[0];
ry(3.077600750942575) q[1];
rz(1.1556665826577452) q[1];
ry(-1.7382343356998118) q[2];
rz(-1.4762801938944818) q[2];
ry(0.4610791386445927) q[3];
rz(-1.3573148291768662) q[3];
ry(-0.04737097465171687) q[4];
rz(0.26393524604900487) q[4];
ry(-3.1320897397606404) q[5];
rz(-1.99170150785179) q[5];
ry(1.7001588898370121) q[6];
rz(1.0773306337478403) q[6];
ry(-1.9036792622259338) q[7];
rz(-2.9556069922868264) q[7];
ry(-3.138767581991448) q[8];
rz(0.6993607024354496) q[8];
ry(3.0781106318563722) q[9];
rz(-0.9510160960408545) q[9];
ry(0.015868949738822913) q[10];
rz(-1.7583141262039426) q[10];
ry(0.703069223869786) q[11];
rz(-0.10316580084265511) q[11];
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
ry(0.24764263376294685) q[0];
rz(-0.9878042399926833) q[0];
ry(-2.887492824076648) q[1];
rz(2.2411855612074887) q[1];
ry(-1.5435340296565956) q[2];
rz(-3.036657048543844) q[2];
ry(1.6298704425917032) q[3];
rz(-2.055448723032867) q[3];
ry(-0.0011662518122609374) q[4];
rz(0.4921559677567765) q[4];
ry(-3.1401190571828885) q[5];
rz(-1.141472461923641) q[5];
ry(0.0040025719751835) q[6];
rz(1.9953383373468196) q[6];
ry(3.1360879993524127) q[7];
rz(2.0003210417545594) q[7];
ry(-3.1414636014165387) q[8];
rz(2.4614525402594865) q[8];
ry(3.1405113298293768) q[9];
rz(0.07408927400120854) q[9];
ry(-0.9976203196270752) q[10];
rz(-0.36539247968078653) q[10];
ry(-1.2318576090872035) q[11];
rz(-3.0948201317929436) q[11];
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
ry(-2.812385012001413) q[0];
rz(-1.2930478577895492) q[0];
ry(2.8195663651909224) q[1];
rz(1.9219754342654576) q[1];
ry(-1.0381890730733705) q[2];
rz(-2.0329618000326444) q[2];
ry(0.2832093097705304) q[3];
rz(-2.027764682696599) q[3];
ry(-2.530440893240692) q[4];
rz(2.7838705671199655) q[4];
ry(0.6152467836976664) q[5];
rz(2.8355701838242915) q[5];
ry(-0.2652081794334853) q[6];
rz(0.5746856361832579) q[6];
ry(0.3150920102086644) q[7];
rz(-1.1333741270645397) q[7];
ry(0.041216005543060596) q[8];
rz(-0.2307565606853785) q[8];
ry(0.07821526479392224) q[9];
rz(0.5856188610649087) q[9];
ry(1.5471933566573675) q[10];
rz(3.022584918210684) q[10];
ry(0.07395644910733701) q[11];
rz(-1.5146911449790061) q[11];