OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.8435648427316713) q[0];
rz(-2.446248758024815) q[0];
ry(2.8132325709298995) q[1];
rz(-2.7544853897882775) q[1];
ry(3.1299304084637294) q[2];
rz(1.1195939927374814) q[2];
ry(0.008838908663474143) q[3];
rz(-2.760275874309744) q[3];
ry(-1.6193337425377399) q[4];
rz(2.0046016724286577) q[4];
ry(-0.2230498451118833) q[5];
rz(-2.4749788860214568) q[5];
ry(-1.7977122982893048) q[6];
rz(2.578069213289029) q[6];
ry(-1.5756198509205073) q[7];
rz(-0.2961120153352441) q[7];
ry(-3.0402560416815425) q[8];
rz(2.8339928809494843) q[8];
ry(-0.03875524065955105) q[9];
rz(0.695371460528344) q[9];
ry(1.7937712876867111) q[10];
rz(0.7974842219577828) q[10];
ry(1.9522054264748485) q[11];
rz(-1.2893928168403193) q[11];
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
ry(-1.7742590386142378) q[0];
rz(-1.861779535978638) q[0];
ry(2.829418357485422) q[1];
rz(-1.2254702212541961) q[1];
ry(-0.08603906903144493) q[2];
rz(-0.19515780740166822) q[2];
ry(-3.0613136400847383) q[3];
rz(-1.0473711464051134) q[3];
ry(3.1297972973030093) q[4];
rz(-2.6911671567132087) q[4];
ry(-0.009275514239274862) q[5];
rz(-0.7701945793444205) q[5];
ry(-3.1399434454049957) q[6];
rz(1.9014238023087469) q[6];
ry(3.135518707096679) q[7];
rz(-2.0329487886561677) q[7];
ry(0.6476115985040796) q[8];
rz(-2.7515023281947877) q[8];
ry(2.960546165520342) q[9];
rz(2.003582985918267) q[9];
ry(-1.8604029263221404) q[10];
rz(3.08494518461703) q[10];
ry(-2.2180771273390842) q[11];
rz(-0.9005127248355436) q[11];
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
ry(2.6778108246757255) q[0];
rz(-2.307092161868574) q[0];
ry(1.446763927181836) q[1];
rz(-2.129105446385201) q[1];
ry(0.013827810505009806) q[2];
rz(2.7657926024389448) q[2];
ry(3.1304142730769975) q[3];
rz(-0.4765431864220404) q[3];
ry(-1.995162691464641) q[4];
rz(0.5765713930255975) q[4];
ry(-1.795489511486167) q[5];
rz(0.004359750968148113) q[5];
ry(0.2239799543789589) q[6];
rz(-2.8751607969348023) q[6];
ry(-2.713296665345854) q[7];
rz(-2.0962351600755786) q[7];
ry(0.32169192771734945) q[8];
rz(0.7619504931138569) q[8];
ry(-3.1064055181538914) q[9];
rz(-0.03265778345991537) q[9];
ry(2.803217768686105) q[10];
rz(-1.1786869977300107) q[10];
ry(-2.5177851614490767) q[11];
rz(0.9228086300794215) q[11];
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
ry(-0.785938652269869) q[0];
rz(-1.152145182469496) q[0];
ry(-1.9867152903231933) q[1];
rz(-0.731678699137043) q[1];
ry(1.569086921508485) q[2];
rz(0.0032759965483483153) q[2];
ry(1.5783513801417257) q[3];
rz(0.00632124183271666) q[3];
ry(3.0179985605377007) q[4];
rz(-1.2058112852570284) q[4];
ry(-1.4758952352164743) q[5];
rz(-0.8892757387239049) q[5];
ry(-3.1175230449910827) q[6];
rz(-0.5716362274933423) q[6];
ry(-0.015697247785725177) q[7];
rz(-0.13351563600870356) q[7];
ry(-1.450302177073656) q[8];
rz(-2.6573509472174393) q[8];
ry(-1.496259240054921) q[9];
rz(-1.650386726731346) q[9];
ry(-2.692317406618693) q[10];
rz(2.853144529556543) q[10];
ry(0.4077331107628622) q[11];
rz(-0.6607672909285025) q[11];
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
ry(3.141304670180947) q[0];
rz(-0.24941591801071764) q[0];
ry(1.5619912041252277) q[1];
rz(1.5853230587246694) q[1];
ry(2.363120587248003) q[2];
rz(1.4858041197132534) q[2];
ry(2.3595742227072023) q[3];
rz(1.6392132767069088) q[3];
ry(-3.0177915081887527) q[4];
rz(-0.18207827802040502) q[4];
ry(-3.1249902733272075) q[5];
rz(2.8954030915562994) q[5];
ry(0.0025447937025003344) q[6];
rz(-1.411052949262022) q[6];
ry(3.141223987509919) q[7];
rz(-2.05524382081348) q[7];
ry(-3.129330388021432) q[8];
rz(0.7107571631126307) q[8];
ry(-2.8891808415717875) q[9];
rz(-0.7944678362193446) q[9];
ry(-1.4775947713293807) q[10];
rz(0.22345265854494453) q[10];
ry(0.007735555122251958) q[11];
rz(0.21870624804366035) q[11];
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
ry(2.9230360005040708) q[0];
rz(2.9532777910201813) q[0];
ry(1.5396179374591823) q[1];
rz(1.042425688329975) q[1];
ry(2.489664751125569) q[2];
rz(-1.44237384351018) q[2];
ry(-2.228326213883709) q[3];
rz(-1.3239562136279754) q[3];
ry(-1.7928793782420749) q[4];
rz(2.684528649622437) q[4];
ry(-3.059899892513372) q[5];
rz(-2.405309660514144) q[5];
ry(1.5622437246783472) q[6];
rz(-1.5984587039270677) q[6];
ry(1.5768211203932232) q[7];
rz(-0.006892768098020419) q[7];
ry(2.655328257163106) q[8];
rz(-2.1836177224929996) q[8];
ry(-0.4357089551088862) q[9];
rz(-1.8546006813751534) q[9];
ry(1.1182135966057052) q[10];
rz(1.3112379581121392) q[10];
ry(0.7246748778415936) q[11];
rz(-2.0747266452124355) q[11];
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
ry(0.7475162945763145) q[0];
rz(-2.845503084027185) q[0];
ry(-1.292521200590514) q[1];
rz(1.9001660962665878) q[1];
ry(1.0082048709615457) q[2];
rz(1.340211785544252) q[2];
ry(1.0369825964966644) q[3];
rz(1.3329283824201845) q[3];
ry(-3.0229165543268324) q[4];
rz(-1.9213796277763215) q[4];
ry(-1.5790188671556187) q[5];
rz(-0.08192372419007275) q[5];
ry(1.5811671282618343) q[6];
rz(2.9964741577498235) q[6];
ry(1.5626052127330077) q[7];
rz(3.109351119595397) q[7];
ry(1.7562843701711333) q[8];
rz(0.9596694610841112) q[8];
ry(1.6381701182958146) q[9];
rz(0.9497259080227374) q[9];
ry(-1.6136028456576783) q[10];
rz(-0.5849678853220283) q[10];
ry(0.008673312733996408) q[11];
rz(2.434703636676827) q[11];
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
ry(-1.7607411092209464) q[0];
rz(-0.4752269134986037) q[0];
ry(0.06939585558968009) q[1];
rz(-3.0804995352944657) q[1];
ry(1.4381116903547888) q[2];
rz(1.0732497817472861) q[2];
ry(1.4363735510903979) q[3];
rz(0.41443279969176405) q[3];
ry(1.530442251460137) q[4];
rz(3.124751705648587) q[4];
ry(-1.7609492687553259) q[5];
rz(-1.8049338286009755) q[5];
ry(-3.139603293843861) q[6];
rz(-0.13926041774872822) q[6];
ry(0.0018458087067854125) q[7];
rz(-3.0884754147535367) q[7];
ry(-1.5685088465792578) q[8];
rz(1.5252763286982054) q[8];
ry(1.5688922747318725) q[9];
rz(2.577582708042284) q[9];
ry(0.603511602755745) q[10];
rz(-1.4123448901961222) q[10];
ry(-2.1176278751773774) q[11];
rz(1.9077228192492) q[11];
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
ry(-1.4151668498256846) q[0];
rz(0.32671640500363186) q[0];
ry(2.47809594000341) q[1];
rz(-2.1618264014198507) q[1];
ry(-3.1289165668257373) q[2];
rz(2.705979041996516) q[2];
ry(0.008809990999606563) q[3];
rz(-2.2860919770576484) q[3];
ry(-1.3853469540669312) q[4];
rz(3.032638312256921) q[4];
ry(1.4590561515971974) q[5];
rz(-1.6541499494012333) q[5];
ry(1.5755886440008096) q[6];
rz(-1.4325093768933375) q[6];
ry(-1.5430822823161405) q[7];
rz(-2.195915082179181) q[7];
ry(-3.0271158031158225) q[8];
rz(3.0972780755431883) q[8];
ry(0.008523824353093673) q[9];
rz(-0.998843305348192) q[9];
ry(-0.8766414631331804) q[10];
rz(-2.354093719658512) q[10];
ry(-0.35544278436350574) q[11];
rz(-1.4090882839195822) q[11];
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
ry(-0.055740832829096476) q[0];
rz(0.08164540297391909) q[0];
ry(0.22997997933573783) q[1];
rz(0.7355210200619263) q[1];
ry(-3.113679805119017) q[2];
rz(-1.4692864248369193) q[2];
ry(3.125389628216686) q[3];
rz(-1.8920234571290524) q[3];
ry(-1.6989817784023806) q[4];
rz(-2.0557820348078835) q[4];
ry(-1.6024573776429432) q[5];
rz(-1.838655843060045) q[5];
ry(-1.4927053021247323) q[6];
rz(-0.5563918308952093) q[6];
ry(-3.13933854177965) q[7];
rz(2.4639649286192844) q[7];
ry(-1.5737928624566022) q[8];
rz(1.6003026857550053) q[8];
ry(1.576873546750751) q[9];
rz(-1.4875816896529213) q[9];
ry(2.095443656289774) q[10];
rz(0.6687527245387566) q[10];
ry(2.0371944197889578) q[11];
rz(0.9251500699458297) q[11];
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
ry(-1.120823406505683) q[0];
rz(-2.7977794116421277) q[0];
ry(1.9866785573128733) q[1];
rz(-1.249022315414538) q[1];
ry(-1.563681211562165) q[2];
rz(-2.5508990750448106) q[2];
ry(1.5800164847161025) q[3];
rz(2.566276135951932) q[3];
ry(3.1381394395487567) q[4];
rz(1.0536290141835194) q[4];
ry(-3.1411729096523677) q[5];
rz(2.4879910276528823) q[5];
ry(-0.001304300493213617) q[6];
rz(-1.0218624475624232) q[6];
ry(3.139334544469133) q[7];
rz(-1.7095972141388442) q[7];
ry(-1.5622846051623078) q[8];
rz(-3.1387357139228746) q[8];
ry(-1.5647357692810346) q[9];
rz(-1.8827286344567988) q[9];
ry(-1.5072002627680723) q[10];
rz(1.4574918268554429) q[10];
ry(0.14803644229690868) q[11];
rz(1.3013403319550543) q[11];
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
ry(-1.5274455168806094) q[0];
rz(-1.0451294978918988) q[0];
ry(1.6284938973297671) q[1];
rz(-0.31804255510723856) q[1];
ry(-1.1767113806906764) q[2];
rz(1.554053156564919) q[2];
ry(-1.594659695466909) q[3];
rz(1.5846131216822994) q[3];
ry(1.4673355890355149) q[4];
rz(0.21793367068793473) q[4];
ry(-0.10563262545893544) q[5];
rz(-0.755496963509736) q[5];
ry(-1.3259804296093014) q[6];
rz(1.3056847178753035) q[6];
ry(-1.5574749801803973) q[7];
rz(-1.567596555822991) q[7];
ry(-0.33405846737286105) q[8];
rz(-1.575623486933) q[8];
ry(3.141082475928981) q[9];
rz(-2.3586479421673174) q[9];
ry(2.0547083512779007) q[10];
rz(2.6446158161969544) q[10];
ry(-1.9353275054700347) q[11];
rz(-2.175168145729556) q[11];
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
ry(-0.0770028281159192) q[0];
rz(-0.3916639902452097) q[0];
ry(-0.17182884080150185) q[1];
rz(0.07555505677730955) q[1];
ry(1.662242911071192) q[2];
rz(3.0911796915577234) q[2];
ry(-1.4850665982372817) q[3];
rz(3.0796421222345716) q[3];
ry(-1.7480517875388424) q[4];
rz(1.6932233548108753) q[4];
ry(0.16280659880112847) q[5];
rz(-2.3740635968913577) q[5];
ry(1.567304371809322) q[6];
rz(-1.9052357128413777) q[6];
ry(-1.5692780072465262) q[7];
rz(-1.7621619662436547) q[7];
ry(-1.5737062123617123) q[8];
rz(-3.1326654047266724) q[8];
ry(-0.0011768703678718866) q[9];
rz(-2.9225979927170966) q[9];
ry(1.234621190160726) q[10];
rz(3.1265243481518628) q[10];
ry(1.5782409890214848) q[11];
rz(-0.2949009192712122) q[11];
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
ry(1.3119622750723263) q[0];
rz(-1.8562094093485986) q[0];
ry(0.26228091298736267) q[1];
rz(-3.0066215101375318) q[1];
ry(1.5874628845099172) q[2];
rz(2.1492940463216064) q[2];
ry(-1.5346658399432305) q[3];
rz(-0.9920729071462886) q[3];
ry(3.132013583767944) q[4];
rz(1.7035377868810162) q[4];
ry(-3.139972282285778) q[5];
rz(-1.1588060654338284) q[5];
ry(-0.0005014387069790019) q[6];
rz(-1.2558606724471122) q[6];
ry(-8.70050793277955e-05) q[7];
rz(0.9713365919635546) q[7];
ry(0.45552619114816517) q[8];
rz(-1.4123394704383423) q[8];
ry(-0.009547079173840345) q[9];
rz(2.9651927616089426) q[9];
ry(-1.5674905133618553) q[10];
rz(-3.111641908043287) q[10];
ry(-1.6185027706170914) q[11];
rz(1.5760804453814168) q[11];
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
ry(-0.3439472167727951) q[0];
rz(-2.7980237694638417) q[0];
ry(0.12786440406201824) q[1];
rz(-2.4379625466665162) q[1];
ry(-0.0029577459674650086) q[2];
rz(-0.6462377826472085) q[2];
ry(-0.011530371254862002) q[3];
rz(0.4599529803445977) q[3];
ry(1.7794962559100136) q[4];
rz(0.5322347917629733) q[4];
ry(0.2705905052619386) q[5];
rz(0.07603336748800578) q[5];
ry(-1.5388055792623079) q[6];
rz(2.2489410345279905) q[6];
ry(1.6021893567358354) q[7];
rz(-1.0223807409561771) q[7];
ry(-0.508188280010028) q[8];
rz(-1.6576941803530048) q[8];
ry(-2.511634451208293) q[9];
rz(2.3631189267212624) q[9];
ry(1.91489597991408) q[10];
rz(1.7186848482457355) q[10];
ry(1.5211912939382881) q[11];
rz(1.5759871119243938) q[11];
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
ry(-1.1286114903694393) q[0];
rz(-0.8251931648456382) q[0];
ry(0.603604641957801) q[1];
rz(-2.3623540668670597) q[1];
ry(3.1093369402733924) q[2];
rz(-0.7369040052314996) q[2];
ry(-1.5849195020032771) q[3];
rz(0.44755127284311613) q[3];
ry(-3.1341602236097605) q[4];
rz(0.3848115867980028) q[4];
ry(-3.119131318504882) q[5];
rz(2.7612382166427603) q[5];
ry(3.1402146439286183) q[6];
rz(1.5954995333256794) q[6];
ry(3.1407941737875262) q[7];
rz(-2.5644184826079917) q[7];
ry(0.020390102259132622) q[8];
rz(2.1028241528606264) q[8];
ry(-0.009406863352447559) q[9];
rz(-1.679468125206828) q[9];
ry(3.0575328387719414) q[10];
rz(1.6305397632098417) q[10];
ry(-0.017119114651827017) q[11];
rz(1.0418094867895924) q[11];
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
ry(-0.05262957367308982) q[0];
rz(-1.7250337297702085) q[0];
ry(3.0977684493083655) q[1];
rz(-1.9031545320118175) q[1];
ry(-0.00634841835489682) q[2];
rz(-1.101984878044465) q[2];
ry(0.025596159518810564) q[3];
rz(2.6732504800805152) q[3];
ry(-3.131931227762771) q[4];
rz(1.4343620636092238) q[4];
ry(-1.5715827387619798) q[5];
rz(0.12320058557935454) q[5];
ry(0.037717980108837416) q[6];
rz(-2.2825669491437157) q[6];
ry(-2.36300232918323) q[7];
rz(0.6981345483760555) q[7];
ry(2.902003366131594) q[8];
rz(-0.7263628789801215) q[8];
ry(2.4701717223594737) q[9];
rz(-1.311507667074313) q[9];
ry(-0.35540642615485396) q[10];
rz(-1.1109668214567687) q[10];
ry(-3.1039759255604524) q[11];
rz(-1.8752362964162133) q[11];
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
ry(1.0549788765973842) q[0];
rz(-0.02266828298781043) q[0];
ry(1.023769605331883) q[1];
rz(0.5951256886660242) q[1];
ry(3.1313646966404565) q[2];
rz(-2.728376407527783) q[2];
ry(2.786853157977398) q[3];
rz(3.0814439964545) q[3];
ry(1.5741571827312086) q[4];
rz(-1.1760129562619603) q[4];
ry(-3.1354008907464266) q[5];
rz(1.7019894525386121) q[5];
ry(-0.0014611856110988697) q[6];
rz(-2.981016956356834) q[6];
ry(3.141289475380579) q[7];
rz(-0.9166900347675863) q[7];
ry(3.128924949295854) q[8];
rz(0.8481872438300675) q[8];
ry(-1.5747587079508387) q[9];
rz(0.2243424311785711) q[9];
ry(-0.01673978008493382) q[10];
rz(-1.0294766811441356) q[10];
ry(-0.011183906490145612) q[11];
rz(-1.6664061860031607) q[11];
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
ry(-2.579476857790692) q[0];
rz(-1.1029641983848721) q[0];
ry(1.8546547566858331) q[1];
rz(-0.003012336319448701) q[1];
ry(-1.5708510297285192) q[2];
rz(-1.5751686310529918) q[2];
ry(-1.5718980774223263) q[3];
rz(-1.5762926527701353) q[3];
ry(1.580812865090345) q[4];
rz(0.03229252327449184) q[4];
ry(-1.5688731325621912) q[5];
rz(-3.1405081905175374) q[5];
ry(-1.571026438532745) q[6];
rz(-0.8473043355261075) q[6];
ry(-1.5706262509811892) q[7];
rz(2.3983058524550436) q[7];
ry(1.7216572576654696) q[8];
rz(-0.4128567345178924) q[8];
ry(1.3254481249016088) q[9];
rz(-1.0631203698064173) q[9];
ry(-1.3924571473528218) q[10];
rz(-3.1118536828128214) q[10];
ry(1.3101725192307134) q[11];
rz(0.7500921690794753) q[11];
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
ry(1.4808053758044624) q[0];
rz(0.013074165952957628) q[0];
ry(3.095222897651104) q[1];
rz(-1.9993648784029474) q[1];
ry(-1.5967880646133459) q[2];
rz(-3.138640188595557) q[2];
ry(-1.5762855950653696) q[3];
rz(1.5781179835774726) q[3];
ry(1.5711424188907168) q[4];
rz(1.5688545153905995) q[4];
ry(1.5719146347348576) q[5];
rz(-3.14105597991659) q[5];
ry(0.0004606161111199538) q[6];
rz(-1.9269407017066928) q[6];
ry(0.00044015172066673924) q[7];
rz(-0.5497869713064) q[7];
ry(-0.015414793943211475) q[8];
rz(-1.5251192991011033) q[8];
ry(-0.008523210708785811) q[9];
rz(1.2406250775731429) q[9];
ry(3.1179122209405272) q[10];
rz(-0.42986288750148965) q[10];
ry(3.134547332423433) q[11];
rz(0.5579379802860928) q[11];
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
ry(1.4756769882372736) q[0];
rz(-1.4882221924929553) q[0];
ry(-2.812677146654335) q[1];
rz(-0.8015054677804425) q[1];
ry(1.563535513408344) q[2];
rz(0.005776109141578446) q[2];
ry(-1.5670548911863285) q[3];
rz(0.00019560565173915023) q[3];
ry(0.208074968178402) q[4];
rz(2.4161534647205802) q[4];
ry(1.5695090354161083) q[5];
rz(-0.28403832855334876) q[5];
ry(-0.3640991577703456) q[6];
rz(-1.930129817553646) q[6];
ry(0.0032006044330762443) q[7];
rz(3.0493211438292263) q[7];
ry(0.49421529635925054) q[8];
rz(-2.4178786520907707) q[8];
ry(-2.9350366652610416) q[9];
rz(0.6230090864706556) q[9];
ry(0.05893661305058676) q[10];
rz(1.7394483978446784) q[10];
ry(-1.8578457679595959) q[11];
rz(2.4413637514235687) q[11];
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
ry(0.03312998173205184) q[0];
rz(0.06277681518024285) q[0];
ry(1.336836330914183) q[1];
rz(0.0075013731015828155) q[1];
ry(-1.5773462876667335) q[2];
rz(0.00013309866421031357) q[2];
ry(1.5710781556168367) q[3];
rz(1.4943898628823886) q[3];
ry(-0.0006721764754696851) q[4];
rz(-0.5554328813063641) q[4];
ry(3.1414612296024953) q[5];
rz(-0.6469305650244195) q[5];
ry(0.0033687884945594604) q[6];
rz(-1.6020013543326386) q[6];
ry(3.140641985567542) q[7];
rz(1.7565829974356588) q[7];
ry(-0.003913664680498919) q[8];
rz(3.0527076977543772) q[8];
ry(-0.01671654840412247) q[9];
rz(2.8739545855168394) q[9];
ry(-3.13298812010696) q[10];
rz(1.2810742727661633) q[10];
ry(0.0199466536950883) q[11];
rz(-0.7620904187601294) q[11];
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
ry(-0.026180079207541947) q[0];
rz(0.23174208021279072) q[0];
ry(0.07033092587612089) q[1];
rz(-0.35061732789616595) q[1];
ry(-1.571093700046391) q[2];
rz(0.006827099757927785) q[2];
ry(-3.115131263926833) q[3];
rz(1.4947922119072792) q[3];
ry(-3.130485337237944) q[4];
rz(-2.852102798166352) q[4];
ry(-0.0001496787219525715) q[5];
rz(-0.7822257361409155) q[5];
ry(-1.9096475616195492) q[6];
rz(0.9278604279522904) q[6];
ry(1.5717372993667176) q[7];
rz(1.5686617365232958) q[7];
ry(-3.0010654436860835) q[8];
rz(-2.901321051365325) q[8];
ry(2.1927257234434503) q[9];
rz(0.3137401874232851) q[9];
ry(-1.8053315545213113) q[10];
rz(-1.876633991597933) q[10];
ry(0.19181356931820837) q[11];
rz(0.2911222640427827) q[11];
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
ry(1.5709398294564245) q[0];
rz(2.885702379958515) q[0];
ry(0.0003879266545006981) q[1];
rz(-1.2221947686537826) q[1];
ry(-1.5641799770191473) q[2];
rz(3.141314186837978) q[2];
ry(-1.5852873633287752) q[3];
rz(1.568552874920637) q[3];
ry(1.5504933630987334) q[4];
rz(-1.7307435954921107) q[4];
ry(3.141307625792377) q[5];
rz(-3.050475615240289) q[5];
ry(-3.1399952994916003) q[6];
rz(-0.40088987670472165) q[6];
ry(0.4256695805786457) q[7];
rz(2.2000158320883765) q[7];
ry(-1.9356065989793416) q[8];
rz(-3.1381214826015977) q[8];
ry(1.5176656384687908) q[9];
rz(-3.1062551858462766) q[9];
ry(3.1415749733774603) q[10];
rz(-2.7239028058966452) q[10];
ry(0.0011631111855723445) q[11];
rz(-2.1466884497959065) q[11];
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
ry(3.140903629825786) q[0];
rz(-1.8264287447041128) q[0];
ry(-1.5616125235984404) q[1];
rz(0.008500990559957703) q[1];
ry(1.5708390311088292) q[2];
rz(-1.5698357481835743) q[2];
ry(2.6382866665786873) q[3];
rz(-1.5732732311516762) q[3];
ry(0.00025518087552645747) q[4];
rz(-0.2956525459884506) q[4];
ry(0.003826837471677003) q[5];
rz(-1.2421066273099453) q[5];
ry(-0.00037163441876527455) q[6];
rz(1.5470744838570918) q[6];
ry(3.1410705685812697) q[7];
rz(2.201219211767836) q[7];
ry(1.3017982286890442) q[8];
rz(0.01065187984151539) q[8];
ry(3.049956011356586) q[9];
rz(-1.9681925541961345) q[9];
ry(1.5710241073395537) q[10];
rz(-2.902264350926942) q[10];
ry(1.5728882749426427) q[11];
rz(-2.523874856685942) q[11];
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
ry(-1.5632387967110681) q[0];
rz(-2.796011049032283) q[0];
ry(-1.57027234112677) q[1];
rz(2.198259524674362) q[1];
ry(-1.5653335942119229) q[2];
rz(1.9163097777302265) q[2];
ry(-1.5757149315609864) q[3];
rz(-0.9432115301759758) q[3];
ry(-0.002236132345435965) q[4];
rz(2.372204426847191) q[4];
ry(1.571261620657486) q[5];
rz(-0.9435399714205818) q[5];
ry(-1.5707906564364198) q[6];
rz(-1.2245680037518372) q[6];
ry(1.6908924567207384) q[7];
rz(2.197712897872943) q[7];
ry(1.551939341868183) q[8];
rz(1.9174185587906907) q[8];
ry(-0.00017861242108363484) q[9];
rz(2.6293149183545848) q[9];
ry(-0.005955950517372521) q[10];
rz(-3.0338042467665542) q[10];
ry(3.1405499833173205) q[11];
rz(-0.32430012520218127) q[11];