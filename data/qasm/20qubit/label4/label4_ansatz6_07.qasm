OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.750110620718119) q[0];
ry(-2.2188586856179198) q[1];
cx q[0],q[1];
ry(-2.0404690094615487) q[0];
ry(0.48848648313754683) q[1];
cx q[0],q[1];
ry(-2.876575115035449) q[1];
ry(1.9155662343056492) q[2];
cx q[1],q[2];
ry(-1.9083333451675317) q[1];
ry(-2.4625462805898577) q[2];
cx q[1],q[2];
ry(2.956484132280141) q[2];
ry(0.4623865757928671) q[3];
cx q[2],q[3];
ry(2.7269935977861324) q[2];
ry(-1.1469145786834385) q[3];
cx q[2],q[3];
ry(0.7323168561287154) q[3];
ry(0.5369147485758914) q[4];
cx q[3],q[4];
ry(-0.921154133411579) q[3];
ry(0.39323722426988367) q[4];
cx q[3],q[4];
ry(2.5799857002710125) q[4];
ry(-0.5960227091574237) q[5];
cx q[4],q[5];
ry(-0.9636680072944683) q[4];
ry(-3.0348735454994284) q[5];
cx q[4],q[5];
ry(-2.1957314082621595) q[5];
ry(-3.112751953889308) q[6];
cx q[5],q[6];
ry(-1.4763219817197948) q[5];
ry(1.5968914859157453) q[6];
cx q[5],q[6];
ry(-0.6859771955071583) q[6];
ry(0.12001172274198613) q[7];
cx q[6],q[7];
ry(1.5189579352678253) q[6];
ry(-2.9677691351823343) q[7];
cx q[6],q[7];
ry(2.657741945614205) q[7];
ry(2.639436751383799) q[8];
cx q[7],q[8];
ry(1.2785624059764114) q[7];
ry(2.9656536232352084) q[8];
cx q[7],q[8];
ry(2.865019701738621) q[8];
ry(1.0461923555644121) q[9];
cx q[8],q[9];
ry(-1.9230994560015895) q[8];
ry(-0.20328436199312439) q[9];
cx q[8],q[9];
ry(1.6887721712062254) q[9];
ry(-0.584810176106247) q[10];
cx q[9],q[10];
ry(-2.9804347004849645) q[9];
ry(-2.950246259350688) q[10];
cx q[9],q[10];
ry(-3.096226817087115) q[10];
ry(-3.1124958223963595) q[11];
cx q[10],q[11];
ry(1.4071902495085078) q[10];
ry(-1.6927448131526364) q[11];
cx q[10],q[11];
ry(1.197017653642194) q[11];
ry(-3.104602334903001) q[12];
cx q[11],q[12];
ry(-1.6033419544501895) q[11];
ry(1.5636452361300777) q[12];
cx q[11],q[12];
ry(-2.8354929699646574) q[12];
ry(2.1338984284207965) q[13];
cx q[12],q[13];
ry(0.3134448651238824) q[12];
ry(-3.102535940768014) q[13];
cx q[12],q[13];
ry(-0.1267971926386986) q[13];
ry(2.7046982213841844) q[14];
cx q[13],q[14];
ry(-1.725889809131898) q[13];
ry(-1.376983466090889) q[14];
cx q[13],q[14];
ry(2.219500359906811) q[14];
ry(-2.775340021566476) q[15];
cx q[14],q[15];
ry(2.046723048763474) q[14];
ry(-2.7260777480159937) q[15];
cx q[14],q[15];
ry(-0.27646705949286154) q[15];
ry(-1.4714432655565484) q[16];
cx q[15],q[16];
ry(-0.10272601885641386) q[15];
ry(2.9196536340730392) q[16];
cx q[15],q[16];
ry(2.4963211501151408) q[16];
ry(-0.48463045513485203) q[17];
cx q[16],q[17];
ry(3.1234478544498776) q[16];
ry(3.120154880273223) q[17];
cx q[16],q[17];
ry(-0.7608730066376959) q[17];
ry(1.9794458161952695) q[18];
cx q[17],q[18];
ry(-1.7917248746919572) q[17];
ry(-0.3795650299609044) q[18];
cx q[17],q[18];
ry(-1.9703333392565003) q[18];
ry(-1.1658597044010195) q[19];
cx q[18],q[19];
ry(-1.0667184660214) q[18];
ry(-0.5657669244416166) q[19];
cx q[18],q[19];
ry(1.4802747865050883) q[0];
ry(0.7089866689104989) q[1];
cx q[0],q[1];
ry(-0.14015362336868545) q[0];
ry(-2.823379847337542) q[1];
cx q[0],q[1];
ry(1.4358329221426818) q[1];
ry(-2.223212227934301) q[2];
cx q[1],q[2];
ry(3.021388434219921) q[1];
ry(1.5566164556405109) q[2];
cx q[1],q[2];
ry(0.2578545310825575) q[2];
ry(2.073373515620687) q[3];
cx q[2],q[3];
ry(1.0204203047868443) q[2];
ry(0.16899038058394697) q[3];
cx q[2],q[3];
ry(0.4088603006343081) q[3];
ry(-2.4581211879709564) q[4];
cx q[3],q[4];
ry(0.16010689780964604) q[3];
ry(-2.036174386653775) q[4];
cx q[3],q[4];
ry(1.5871034793817476) q[4];
ry(1.6128068693386357) q[5];
cx q[4],q[5];
ry(-1.4145694615897586) q[4];
ry(-1.3042126356119321) q[5];
cx q[4],q[5];
ry(1.423307955582457) q[5];
ry(1.5943019948121) q[6];
cx q[5],q[6];
ry(-0.12524722456966764) q[5];
ry(1.472268143013597) q[6];
cx q[5],q[6];
ry(1.4521471347376937) q[6];
ry(-2.389792260111911) q[7];
cx q[6],q[7];
ry(-0.2185757811911702) q[6];
ry(1.5719374900099057) q[7];
cx q[6],q[7];
ry(0.6440993409121144) q[7];
ry(1.9417238494364266) q[8];
cx q[7],q[8];
ry(1.5238362406168333) q[7];
ry(2.0796126470066487) q[8];
cx q[7],q[8];
ry(-1.694779361590705) q[8];
ry(0.23136140051654208) q[9];
cx q[8],q[9];
ry(-0.008808831409368167) q[8];
ry(1.4054181862159536) q[9];
cx q[8],q[9];
ry(2.7756283086479865) q[9];
ry(1.7296173698839334) q[10];
cx q[9],q[10];
ry(1.992610565457026) q[9];
ry(-1.1002051954495113) q[10];
cx q[9],q[10];
ry(0.546669343072681) q[10];
ry(1.44228854553585) q[11];
cx q[10],q[11];
ry(-1.593133111462245) q[10];
ry(-1.4915229974650872) q[11];
cx q[10],q[11];
ry(0.46663502892914455) q[11];
ry(-0.3501456660153872) q[12];
cx q[11],q[12];
ry(0.16217165356577645) q[11];
ry(0.029899508086809412) q[12];
cx q[11],q[12];
ry(-0.04437846970341404) q[12];
ry(-2.8536450938834204) q[13];
cx q[12],q[13];
ry(2.3014002960294047) q[12];
ry(0.007536586886880414) q[13];
cx q[12],q[13];
ry(-1.3806979953789273) q[13];
ry(-0.7762794732960299) q[14];
cx q[13],q[14];
ry(-0.9990104726062361) q[13];
ry(-1.9576937158544716) q[14];
cx q[13],q[14];
ry(1.1201495741112408) q[14];
ry(-2.634663239090304) q[15];
cx q[14],q[15];
ry(3.084614486797799) q[14];
ry(-0.09491225823239358) q[15];
cx q[14],q[15];
ry(-0.5027497221180841) q[15];
ry(-0.9660554628526007) q[16];
cx q[15],q[16];
ry(2.5182122389569854) q[15];
ry(-3.106655474207167) q[16];
cx q[15],q[16];
ry(3.073073266318313) q[16];
ry(-1.1414026832260735) q[17];
cx q[16],q[17];
ry(0.8475780985834422) q[16];
ry(-3.008773470186258) q[17];
cx q[16],q[17];
ry(-2.9722081069705872) q[17];
ry(-0.35171795123679317) q[18];
cx q[17],q[18];
ry(-1.2185689312428227) q[17];
ry(-1.3953958006277452) q[18];
cx q[17],q[18];
ry(2.4010376320973856) q[18];
ry(1.0719208515649457) q[19];
cx q[18],q[19];
ry(1.5561253035667473) q[18];
ry(-0.09726311035749641) q[19];
cx q[18],q[19];
ry(1.0511557755528176) q[0];
ry(0.44560055808543547) q[1];
cx q[0],q[1];
ry(1.286707452568937) q[0];
ry(1.24476920044769) q[1];
cx q[0],q[1];
ry(-2.852742618602517) q[1];
ry(-1.8676457831637465) q[2];
cx q[1],q[2];
ry(-2.394175512621695) q[1];
ry(-2.2886010055899813) q[2];
cx q[1],q[2];
ry(1.7969647247393157) q[2];
ry(-0.5934695582698115) q[3];
cx q[2],q[3];
ry(2.7756688051247043) q[2];
ry(1.9910635962679153) q[3];
cx q[2],q[3];
ry(2.277538047744433) q[3];
ry(1.1994093784703823) q[4];
cx q[3],q[4];
ry(0.35275345991308255) q[3];
ry(-3.138052398183743) q[4];
cx q[3],q[4];
ry(-1.6134709543485544) q[4];
ry(2.8013802792300333) q[5];
cx q[4],q[5];
ry(3.078708954787574) q[4];
ry(2.989994532188873) q[5];
cx q[4],q[5];
ry(2.833107013035945) q[5];
ry(-1.4086687754030063) q[6];
cx q[5],q[6];
ry(-0.17370876833604945) q[5];
ry(-1.695684159062913) q[6];
cx q[5],q[6];
ry(1.485067971235166) q[6];
ry(1.6042551458044034) q[7];
cx q[6],q[7];
ry(-1.606422243647655) q[6];
ry(-2.878477545973962) q[7];
cx q[6],q[7];
ry(-2.3583178089547463) q[7];
ry(-1.565718628512538) q[8];
cx q[7],q[8];
ry(1.3803653121460924) q[7];
ry(-0.01901455104166484) q[8];
cx q[7],q[8];
ry(1.4725396989898691) q[8];
ry(-2.011125440888315) q[9];
cx q[8],q[9];
ry(-0.21101290484966562) q[8];
ry(-0.14618947252968706) q[9];
cx q[8],q[9];
ry(-2.1042629821188115) q[9];
ry(1.9121672857952818) q[10];
cx q[9],q[10];
ry(-0.8415764372506995) q[9];
ry(3.1051762701601686) q[10];
cx q[9],q[10];
ry(-3.1398483799279244) q[10];
ry(0.34104218311843404) q[11];
cx q[10],q[11];
ry(3.0289232464171185) q[10];
ry(-3.1190228266003155) q[11];
cx q[10],q[11];
ry(0.4389500278630498) q[11];
ry(0.45216836175212016) q[12];
cx q[11],q[12];
ry(-1.7534314321060016) q[11];
ry(-1.3964981056880905) q[12];
cx q[11],q[12];
ry(-1.6768093831032047) q[12];
ry(2.8600457894921227) q[13];
cx q[12],q[13];
ry(-2.4321744035525583) q[12];
ry(-0.9550159180747402) q[13];
cx q[12],q[13];
ry(1.576492778141874) q[13];
ry(1.7931166294822998) q[14];
cx q[13],q[14];
ry(-1.1568256975798572) q[13];
ry(2.368204681334378) q[14];
cx q[13],q[14];
ry(0.6084680685854935) q[14];
ry(-2.828324231114455) q[15];
cx q[14],q[15];
ry(-2.9561833940095004) q[14];
ry(-0.12774896417244053) q[15];
cx q[14],q[15];
ry(2.7194823644980297) q[15];
ry(2.224126318896733) q[16];
cx q[15],q[16];
ry(-3.0750124113841544) q[15];
ry(0.03141486093478158) q[16];
cx q[15],q[16];
ry(-1.1218025924443928) q[16];
ry(1.3861100147234398) q[17];
cx q[16],q[17];
ry(-0.8220473933102884) q[16];
ry(-0.2114821349906446) q[17];
cx q[16],q[17];
ry(2.4496624746109155) q[17];
ry(1.0542859474300235) q[18];
cx q[17],q[18];
ry(-3.0802659313176255) q[17];
ry(2.092894590980578) q[18];
cx q[17],q[18];
ry(-0.4740513064140277) q[18];
ry(-1.661418469102474) q[19];
cx q[18],q[19];
ry(1.0506910680054142) q[18];
ry(1.6674572890711303) q[19];
cx q[18],q[19];
ry(-2.5131055578971138) q[0];
ry(-2.700581633940562) q[1];
cx q[0],q[1];
ry(-0.691655525778849) q[0];
ry(-1.3453144020107457) q[1];
cx q[0],q[1];
ry(-2.8205417947399996) q[1];
ry(1.3929066864594217) q[2];
cx q[1],q[2];
ry(-1.860110663186) q[1];
ry(0.07820844603853828) q[2];
cx q[1],q[2];
ry(-1.5950468612766726) q[2];
ry(-2.684373731816852) q[3];
cx q[2],q[3];
ry(-2.8750929653721737) q[2];
ry(-2.937621055243156) q[3];
cx q[2],q[3];
ry(-2.5503685740037647) q[3];
ry(1.316979573901213) q[4];
cx q[3],q[4];
ry(-0.36088113640797914) q[3];
ry(-2.958565789754067) q[4];
cx q[3],q[4];
ry(-1.7987364117115439) q[4];
ry(1.5734799975511438) q[5];
cx q[4],q[5];
ry(-0.47108472784655314) q[4];
ry(2.8476105313870574) q[5];
cx q[4],q[5];
ry(-1.580754323803874) q[5];
ry(1.5778909562504921) q[6];
cx q[5],q[6];
ry(-0.22602668522672636) q[5];
ry(-0.15526660833905773) q[6];
cx q[5],q[6];
ry(1.6091738839365144) q[6];
ry(-0.8040533326258704) q[7];
cx q[6],q[7];
ry(-1.176308849239193) q[6];
ry(-2.4209628211920804) q[7];
cx q[6],q[7];
ry(0.30314194711358394) q[7];
ry(1.643619731189312) q[8];
cx q[7],q[8];
ry(-2.1284510048469807) q[7];
ry(3.1378173404567296) q[8];
cx q[7],q[8];
ry(-1.5443690069875375) q[8];
ry(-2.859784660251896) q[9];
cx q[8],q[9];
ry(-2.6713967093017406) q[8];
ry(1.8362627803107987) q[9];
cx q[8],q[9];
ry(1.655229136324456) q[9];
ry(-2.6536257445911753) q[10];
cx q[9],q[10];
ry(2.597742706768193) q[9];
ry(-1.6642052715222864) q[10];
cx q[9],q[10];
ry(-1.4176398274669972) q[10];
ry(-1.5406030976010454) q[11];
cx q[10],q[11];
ry(-0.4416114880083772) q[10];
ry(2.789256680608551) q[11];
cx q[10],q[11];
ry(2.0834836484320896) q[11];
ry(1.6193452177718133) q[12];
cx q[11],q[12];
ry(1.2283449992001332) q[11];
ry(-3.136604440363621) q[12];
cx q[11],q[12];
ry(1.5628615912049026) q[12];
ry(1.6522184172232048) q[13];
cx q[12],q[13];
ry(-2.6123958091636337) q[12];
ry(2.189860987568143) q[13];
cx q[12],q[13];
ry(-0.07996837158344494) q[13];
ry(1.0789124546238442) q[14];
cx q[13],q[14];
ry(-0.20811292269010678) q[13];
ry(3.046533227519597) q[14];
cx q[13],q[14];
ry(-2.861144208576819) q[14];
ry(-0.04579796280978332) q[15];
cx q[14],q[15];
ry(-1.6590384662625342) q[14];
ry(-0.27705467413812546) q[15];
cx q[14],q[15];
ry(3.1228479978279227) q[15];
ry(1.339658455220248) q[16];
cx q[15],q[16];
ry(-1.6722104325059854) q[15];
ry(-3.1380562288578204) q[16];
cx q[15],q[16];
ry(1.468451765293325) q[16];
ry(2.001828425121997) q[17];
cx q[16],q[17];
ry(2.6665943985666516) q[16];
ry(-0.6222088650884349) q[17];
cx q[16],q[17];
ry(-1.6021188219457696) q[17];
ry(-1.741515500023775) q[18];
cx q[17],q[18];
ry(-1.5172476482858164) q[17];
ry(-1.3534228391877057) q[18];
cx q[17],q[18];
ry(-1.5079177664985688) q[18];
ry(-2.6958490657559504) q[19];
cx q[18],q[19];
ry(-1.9178751760689734) q[18];
ry(-1.952894487810771) q[19];
cx q[18],q[19];
ry(-2.765440783257568) q[0];
ry(-2.413450176980747) q[1];
cx q[0],q[1];
ry(-0.3998769779451976) q[0];
ry(2.3348801671051875) q[1];
cx q[0],q[1];
ry(1.4747973867968351) q[1];
ry(0.2743433449153451) q[2];
cx q[1],q[2];
ry(-2.9353126482080683) q[1];
ry(2.0353972645107063) q[2];
cx q[1],q[2];
ry(2.8023930088124427) q[2];
ry(-1.331177277769586) q[3];
cx q[2],q[3];
ry(2.4576965826079316) q[2];
ry(-0.33816233562346554) q[3];
cx q[2],q[3];
ry(1.0949787383908727) q[3];
ry(-1.5837198469677223) q[4];
cx q[3],q[4];
ry(1.2838792902725096) q[3];
ry(2.1397846283878996) q[4];
cx q[3],q[4];
ry(-3.13088378405212) q[4];
ry(-1.5464097149165947) q[5];
cx q[4],q[5];
ry(-1.7661287191243833) q[4];
ry(-0.0022381646612386814) q[5];
cx q[4],q[5];
ry(0.232119461272168) q[5];
ry(1.5923574070621063) q[6];
cx q[5],q[6];
ry(-2.8007009156915705) q[5];
ry(-0.016907941282466865) q[6];
cx q[5],q[6];
ry(-1.5402988914684947) q[6];
ry(0.22526859637360944) q[7];
cx q[6],q[7];
ry(1.654619452785335) q[6];
ry(-2.7463521265330604) q[7];
cx q[6],q[7];
ry(1.7031264568042008) q[7];
ry(0.04480865459892655) q[8];
cx q[7],q[8];
ry(1.8760993375530897) q[7];
ry(1.3647234558971775) q[8];
cx q[7],q[8];
ry(-1.9987446524175532) q[8];
ry(3.095038687887996) q[9];
cx q[8],q[9];
ry(-0.05845405313931007) q[8];
ry(-0.02613439147543417) q[9];
cx q[8],q[9];
ry(-1.8836873018673141) q[9];
ry(1.6585105693443365) q[10];
cx q[9],q[10];
ry(1.104420721705444) q[9];
ry(-0.05525112911005792) q[10];
cx q[9],q[10];
ry(-0.09894973214590053) q[10];
ry(-1.0157575022468266) q[11];
cx q[10],q[11];
ry(-2.8725358289257836) q[10];
ry(-3.1138119530139035) q[11];
cx q[10],q[11];
ry(1.5401727499896924) q[11];
ry(-2.952856889076783) q[12];
cx q[11],q[12];
ry(-0.12581264469285447) q[11];
ry(-1.631988143805198) q[12];
cx q[11],q[12];
ry(2.7269068566108277) q[12];
ry(2.4215605923143433) q[13];
cx q[12],q[13];
ry(0.114132171103112) q[12];
ry(3.1258634077975946) q[13];
cx q[12],q[13];
ry(-3.0376154984836163) q[13];
ry(-1.9059208246845434) q[14];
cx q[13],q[14];
ry(1.3569729731217837) q[13];
ry(-0.07710638326584238) q[14];
cx q[13],q[14];
ry(-1.6584487552189051) q[14];
ry(2.922296845380906) q[15];
cx q[14],q[15];
ry(-0.05169754397749309) q[14];
ry(-0.6348659604637552) q[15];
cx q[14],q[15];
ry(1.8425112541624329) q[15];
ry(-1.9216962623841427) q[16];
cx q[15],q[16];
ry(-0.026649928302254544) q[15];
ry(0.3380250207606119) q[16];
cx q[15],q[16];
ry(1.6762876115535867) q[16];
ry(1.2865170735418543) q[17];
cx q[16],q[17];
ry(-0.8660940548673783) q[16];
ry(2.0946361908755646) q[17];
cx q[16],q[17];
ry(-1.1764161135286377) q[17];
ry(-1.5385297519805365) q[18];
cx q[17],q[18];
ry(-2.183912319761945) q[17];
ry(-2.363074770322859) q[18];
cx q[17],q[18];
ry(-1.5579904276138257) q[18];
ry(-1.8822166361894592) q[19];
cx q[18],q[19];
ry(2.632947577600637) q[18];
ry(-0.2643765884308298) q[19];
cx q[18],q[19];
ry(1.309983275003495) q[0];
ry(-1.7341948138741916) q[1];
cx q[0],q[1];
ry(-1.623825184747006) q[0];
ry(-1.141502540433247) q[1];
cx q[0],q[1];
ry(0.8970594103292888) q[1];
ry(-0.8691032416834572) q[2];
cx q[1],q[2];
ry(1.29018230231209) q[1];
ry(1.8796252974577756) q[2];
cx q[1],q[2];
ry(-2.0850530186884195) q[2];
ry(0.9292970280939841) q[3];
cx q[2],q[3];
ry(-0.008807310543828883) q[2];
ry(2.4705052613338863) q[3];
cx q[2],q[3];
ry(-0.8409677618696465) q[3];
ry(-0.027659586123256478) q[4];
cx q[3],q[4];
ry(1.2169914351270208) q[3];
ry(-2.7970858451857255) q[4];
cx q[3],q[4];
ry(-1.900052292212715) q[4];
ry(-1.8999452065373656) q[5];
cx q[4],q[5];
ry(2.478723221336141) q[4];
ry(0.001468769331298326) q[5];
cx q[4],q[5];
ry(2.65117498413507) q[5];
ry(-0.24437677826659868) q[6];
cx q[5],q[6];
ry(-3.131567742552894) q[5];
ry(3.1275293561395556) q[6];
cx q[5],q[6];
ry(-1.037878560794268) q[6];
ry(-0.39080050729269183) q[7];
cx q[6],q[7];
ry(-3.076667382359219) q[6];
ry(-0.010798164760555375) q[7];
cx q[6],q[7];
ry(-2.6562118529725076) q[7];
ry(-2.614530225204499) q[8];
cx q[7],q[8];
ry(1.5938366190111481) q[7];
ry(0.12040460106087103) q[8];
cx q[7],q[8];
ry(-0.2413125381068344) q[8];
ry(-2.193148619844429) q[9];
cx q[8],q[9];
ry(-0.042043198265759744) q[8];
ry(-0.03183666130479652) q[9];
cx q[8],q[9];
ry(-1.2350093566198765) q[9];
ry(-0.10085115601220451) q[10];
cx q[9],q[10];
ry(1.747733415337163) q[9];
ry(0.008234928250203666) q[10];
cx q[9],q[10];
ry(1.5585835394462728) q[10];
ry(1.433829680527545) q[11];
cx q[10],q[11];
ry(3.12070196308532) q[10];
ry(1.281929394746415) q[11];
cx q[10],q[11];
ry(0.005173375828740667) q[11];
ry(1.3184650691996982) q[12];
cx q[11],q[12];
ry(-1.291966758760367) q[11];
ry(0.0028969944482755143) q[12];
cx q[11],q[12];
ry(-1.4284752637160523) q[12];
ry(-0.04327168707967477) q[13];
cx q[12],q[13];
ry(0.1259340602682544) q[12];
ry(-0.7969643448760283) q[13];
cx q[12],q[13];
ry(-1.1435474421683334) q[13];
ry(1.580549352416068) q[14];
cx q[13],q[14];
ry(2.505919789390749) q[13];
ry(2.7726350251551186) q[14];
cx q[13],q[14];
ry(-0.8568664911772441) q[14];
ry(-1.7556273156085949) q[15];
cx q[14],q[15];
ry(-0.13469011271986026) q[14];
ry(-2.5672652819709407) q[15];
cx q[14],q[15];
ry(2.2945806305397096) q[15];
ry(-0.9672269315250711) q[16];
cx q[15],q[16];
ry(0.04901635476486188) q[15];
ry(0.028708783318875497) q[16];
cx q[15],q[16];
ry(-2.148917152452637) q[16];
ry(1.691047566189841) q[17];
cx q[16],q[17];
ry(0.8057331921405848) q[16];
ry(0.05953884970525447) q[17];
cx q[16],q[17];
ry(-1.0232396888932254) q[17];
ry(1.588054535997932) q[18];
cx q[17],q[18];
ry(-2.4250918029368167) q[17];
ry(-3.071940239936579) q[18];
cx q[17],q[18];
ry(-0.6562399347881298) q[18];
ry(-1.6966220001680823) q[19];
cx q[18],q[19];
ry(0.6606997307411451) q[18];
ry(1.3384459690480615) q[19];
cx q[18],q[19];
ry(1.7205455336662547) q[0];
ry(2.9160759354966093) q[1];
cx q[0],q[1];
ry(-1.739300422476636) q[0];
ry(-1.0540599646663953) q[1];
cx q[0],q[1];
ry(-2.2360162689228877) q[1];
ry(1.872222423090159) q[2];
cx q[1],q[2];
ry(-0.6540176846035172) q[1];
ry(-1.8122091559484816) q[2];
cx q[1],q[2];
ry(-1.821941059455991) q[2];
ry(1.7458934287760375) q[3];
cx q[2],q[3];
ry(-0.06739893326663893) q[2];
ry(-1.402947859534793) q[3];
cx q[2],q[3];
ry(-1.5007304674535884) q[3];
ry(2.6770979222282527) q[4];
cx q[3],q[4];
ry(3.1350204308618737) q[3];
ry(1.7378163391118822) q[4];
cx q[3],q[4];
ry(-0.7986454797893154) q[4];
ry(1.1414038565910571) q[5];
cx q[4],q[5];
ry(-1.6588441225751143) q[4];
ry(2.5195568798866326) q[5];
cx q[4],q[5];
ry(0.4626488340891033) q[5];
ry(2.346512416149634) q[6];
cx q[5],q[6];
ry(1.2767977324539104) q[5];
ry(2.331241308364781) q[6];
cx q[5],q[6];
ry(3.060445744071109) q[6];
ry(-2.4330650095702344) q[7];
cx q[6],q[7];
ry(1.4614916343069653) q[6];
ry(3.1382445797610803) q[7];
cx q[6],q[7];
ry(1.5551156864984383) q[7];
ry(-0.9724442838165084) q[8];
cx q[7],q[8];
ry(1.5801463108184066) q[7];
ry(2.9413792337802978) q[8];
cx q[7],q[8];
ry(1.5699463541387413) q[8];
ry(0.6373996934223118) q[9];
cx q[8],q[9];
ry(1.5978019100907255) q[8];
ry(-1.7964497914304922) q[9];
cx q[8],q[9];
ry(-1.1329620234375666) q[9];
ry(-1.5857859110717554) q[10];
cx q[9],q[10];
ry(-1.5948974531351554) q[9];
ry(3.140746720523432) q[10];
cx q[9],q[10];
ry(2.1358372773849545) q[10];
ry(-0.1400145947170897) q[11];
cx q[10],q[11];
ry(-1.5824931071908965) q[10];
ry(-3.118154575774704) q[11];
cx q[10],q[11];
ry(-1.567176291670897) q[11];
ry(-2.9033467701033415) q[12];
cx q[11],q[12];
ry(0.05500786918058166) q[11];
ry(-2.8484484505441254) q[12];
cx q[11],q[12];
ry(-3.0842846106356143) q[12];
ry(0.18161182191214612) q[13];
cx q[12],q[13];
ry(3.0847618857301247) q[12];
ry(0.007108491823932361) q[13];
cx q[12],q[13];
ry(0.9550631490246883) q[13];
ry(1.9513748274194582) q[14];
cx q[13],q[14];
ry(-0.09311904806710367) q[13];
ry(-0.06613456599349204) q[14];
cx q[13],q[14];
ry(2.2468530719544892) q[14];
ry(0.9885423419500299) q[15];
cx q[14],q[15];
ry(2.7926092888457004) q[14];
ry(2.345196869872331) q[15];
cx q[14],q[15];
ry(-1.5409422033556368) q[15];
ry(1.5473789907718538) q[16];
cx q[15],q[16];
ry(-1.5645052612317216) q[15];
ry(1.8984370348916757) q[16];
cx q[15],q[16];
ry(1.6061469457974287) q[16];
ry(-0.9131106734001033) q[17];
cx q[16],q[17];
ry(1.3192475324017072) q[16];
ry(-0.6563438517945261) q[17];
cx q[16],q[17];
ry(1.4899519114878137) q[17];
ry(0.4238073378557837) q[18];
cx q[17],q[18];
ry(-2.928242785379023) q[17];
ry(-2.778343492186682) q[18];
cx q[17],q[18];
ry(0.8398654558955698) q[18];
ry(-0.7712593356309918) q[19];
cx q[18],q[19];
ry(1.490910469789675) q[18];
ry(1.6361781497466494) q[19];
cx q[18],q[19];
ry(2.660183063142642) q[0];
ry(-1.550383690301362) q[1];
cx q[0],q[1];
ry(-1.870108975061825) q[0];
ry(-2.482881282298772) q[1];
cx q[0],q[1];
ry(0.23044258902472237) q[1];
ry(0.46876400110611116) q[2];
cx q[1],q[2];
ry(-0.8524097544627516) q[1];
ry(-1.4290025184836859) q[2];
cx q[1],q[2];
ry(0.9280202927885947) q[2];
ry(0.4822184714059909) q[3];
cx q[2],q[3];
ry(-3.1022529149448306) q[2];
ry(-3.1363461112441815) q[3];
cx q[2],q[3];
ry(0.8363899834916895) q[3];
ry(-1.5739342622314225) q[4];
cx q[3],q[4];
ry(2.5262731015523388) q[3];
ry(0.004948794812536516) q[4];
cx q[3],q[4];
ry(-2.5130179759734466) q[4];
ry(-2.160429282710624) q[5];
cx q[4],q[5];
ry(0.014659316335161184) q[4];
ry(0.007079519425207793) q[5];
cx q[4],q[5];
ry(-0.7847572518598769) q[5];
ry(-1.483734578020388) q[6];
cx q[5],q[6];
ry(-1.3849325218312787) q[5];
ry(2.333527889367006) q[6];
cx q[5],q[6];
ry(-2.835698520496637) q[6];
ry(2.87723651093645) q[7];
cx q[6],q[7];
ry(-3.124182366212003) q[6];
ry(-3.1374273002565314) q[7];
cx q[6],q[7];
ry(-0.14011059221213623) q[7];
ry(-1.5712632316007988) q[8];
cx q[7],q[8];
ry(0.7100958529658572) q[7];
ry(-0.05511958288058505) q[8];
cx q[7],q[8];
ry(0.009036389434445227) q[8];
ry(2.0095664698835485) q[9];
cx q[8],q[9];
ry(1.1684808084597034) q[8];
ry(3.140446891063882) q[9];
cx q[8],q[9];
ry(-1.5770489711428306) q[9];
ry(2.142320177372954) q[10];
cx q[9],q[10];
ry(-0.5740376306770101) q[9];
ry(2.9065288800596294) q[10];
cx q[9],q[10];
ry(-1.5706493693643262) q[10];
ry(-1.010703913446835) q[11];
cx q[10],q[11];
ry(-3.1407265671254163) q[10];
ry(1.1106649792857253) q[11];
cx q[10],q[11];
ry(0.5032744049369722) q[11];
ry(1.2403496874014854) q[12];
cx q[11],q[12];
ry(1.633442885055919) q[11];
ry(1.5454240457762394) q[12];
cx q[11],q[12];
ry(0.11627152456946366) q[12];
ry(1.0623146834042707) q[13];
cx q[12],q[13];
ry(-3.1070547651216724) q[12];
ry(3.0952367354656514) q[13];
cx q[12],q[13];
ry(-1.402121993393476) q[13];
ry(-1.0639395913750915) q[14];
cx q[13],q[14];
ry(-2.059236692737173) q[13];
ry(2.130221750126247) q[14];
cx q[13],q[14];
ry(-1.6356106171165576) q[14];
ry(1.1259190342056762) q[15];
cx q[14],q[15];
ry(-3.0948089297630266) q[14];
ry(2.9046640225133387) q[15];
cx q[14],q[15];
ry(-1.9586090625112065) q[15];
ry(-1.7903892964181305) q[16];
cx q[15],q[16];
ry(-3.0747276959829457) q[15];
ry(2.7997251008732857) q[16];
cx q[15],q[16];
ry(1.3814100581117374) q[16];
ry(-1.303556952579422) q[17];
cx q[16],q[17];
ry(-2.6530557101679295) q[16];
ry(-0.8156681361350344) q[17];
cx q[16],q[17];
ry(1.8975111104417817) q[17];
ry(1.8240276525839763) q[18];
cx q[17],q[18];
ry(-0.17114825446340182) q[17];
ry(0.547412266998598) q[18];
cx q[17],q[18];
ry(-1.1039852384181457) q[18];
ry(0.7336904707465957) q[19];
cx q[18],q[19];
ry(1.8706951654113615) q[18];
ry(1.204625145296036) q[19];
cx q[18],q[19];
ry(-3.053722839942853) q[0];
ry(-2.6348128113636284) q[1];
cx q[0],q[1];
ry(1.8482989582880984) q[0];
ry(1.3388724417199693) q[1];
cx q[0],q[1];
ry(1.9473668722593342) q[1];
ry(1.5122021126072909) q[2];
cx q[1],q[2];
ry(-1.2800327290815279) q[1];
ry(-2.842050619659899) q[2];
cx q[1],q[2];
ry(0.004332303712979879) q[2];
ry(1.2474450408836244) q[3];
cx q[2],q[3];
ry(2.8143090446238896) q[2];
ry(-2.1391472939961513) q[3];
cx q[2],q[3];
ry(-1.5650861107794733) q[3];
ry(2.5213663752419486) q[4];
cx q[3],q[4];
ry(2.3987428355986666) q[3];
ry(0.9556599244215037) q[4];
cx q[3],q[4];
ry(-0.0068190840944439245) q[4];
ry(-0.10565584130504124) q[5];
cx q[4],q[5];
ry(0.0011684922056520678) q[4];
ry(-0.16816746850102496) q[5];
cx q[4],q[5];
ry(1.83986382301583) q[5];
ry(0.8614282695302888) q[6];
cx q[5],q[6];
ry(-1.3234598326166174) q[5];
ry(0.5827275500350403) q[6];
cx q[5],q[6];
ry(0.05366771537641105) q[6];
ry(-1.6763594611896653) q[7];
cx q[6],q[7];
ry(1.4394518804924106) q[6];
ry(0.34781487643278375) q[7];
cx q[6],q[7];
ry(1.5811156314973867) q[7];
ry(-3.1330079445591474) q[8];
cx q[7],q[8];
ry(0.32848357682775803) q[7];
ry(-2.0483967081377443) q[8];
cx q[7],q[8];
ry(1.5689766948174084) q[8];
ry(1.5684871704552499) q[9];
cx q[8],q[9];
ry(0.6615665907408212) q[8];
ry(2.034673020619211) q[9];
cx q[8],q[9];
ry(-1.57594861745618) q[9];
ry(0.048384530538871116) q[10];
cx q[9],q[10];
ry(-1.5320555464618302) q[9];
ry(1.5011924171585682) q[10];
cx q[9],q[10];
ry(2.339612215180117) q[10];
ry(0.5587404388721597) q[11];
cx q[10],q[11];
ry(-0.015190233634129411) q[10];
ry(-3.1368712109134718) q[11];
cx q[10],q[11];
ry(1.2318693415227897) q[11];
ry(-2.5533333098300974) q[12];
cx q[11],q[12];
ry(-3.0923669644543685) q[11];
ry(1.616439537568849) q[12];
cx q[11],q[12];
ry(2.09914080197024) q[12];
ry(1.5800935649087489) q[13];
cx q[12],q[13];
ry(-1.608047285190924) q[12];
ry(2.714224010057373) q[13];
cx q[12],q[13];
ry(-1.571639952839945) q[13];
ry(1.815861999023411) q[14];
cx q[13],q[14];
ry(-1.5618696866084028) q[13];
ry(-2.0014400186303645) q[14];
cx q[13],q[14];
ry(-1.3035294700832751) q[14];
ry(1.6266720057288115) q[15];
cx q[14],q[15];
ry(0.0009245698910653745) q[14];
ry(-3.138976744089568) q[15];
cx q[14],q[15];
ry(-1.5564065286383102) q[15];
ry(-1.5951183200166836) q[16];
cx q[15],q[16];
ry(-0.4091770005918596) q[15];
ry(1.6022010144453742) q[16];
cx q[15],q[16];
ry(1.579747162611584) q[16];
ry(-1.5815590808560038) q[17];
cx q[16],q[17];
ry(2.845296767382788) q[16];
ry(2.1614729696458825) q[17];
cx q[16],q[17];
ry(-2.4133459421362216) q[17];
ry(2.8254386006789596) q[18];
cx q[17],q[18];
ry(-0.2054063161823365) q[17];
ry(-3.034208880513053) q[18];
cx q[17],q[18];
ry(-1.3491239055774629) q[18];
ry(1.4463331524260807) q[19];
cx q[18],q[19];
ry(-0.1721940039924634) q[18];
ry(-1.5796372225250526) q[19];
cx q[18],q[19];
ry(0.06297993897257115) q[0];
ry(-2.674238227046991) q[1];
cx q[0],q[1];
ry(1.295939897471876) q[0];
ry(-0.317891889361392) q[1];
cx q[0],q[1];
ry(0.918501243620705) q[1];
ry(1.6085007775484952) q[2];
cx q[1],q[2];
ry(2.482655595762646) q[1];
ry(0.04617031464752497) q[2];
cx q[1],q[2];
ry(1.6060673012395963) q[2];
ry(-1.5672109360895459) q[3];
cx q[2],q[3];
ry(-1.788084987481953) q[2];
ry(1.2755888706662732) q[3];
cx q[2],q[3];
ry(-1.4174476921415167) q[3];
ry(-0.7902608633468059) q[4];
cx q[3],q[4];
ry(-0.007963130078063757) q[3];
ry(3.1186352862360915) q[4];
cx q[3],q[4];
ry(-0.7698250605849163) q[4];
ry(3.1009221813487504) q[5];
cx q[4],q[5];
ry(1.4040126541414217) q[4];
ry(1.5410973043224538) q[5];
cx q[4],q[5];
ry(-1.5681484125181553) q[5];
ry(-1.559875758119854) q[6];
cx q[5],q[6];
ry(0.16465537582462186) q[5];
ry(2.6797797878572274) q[6];
cx q[5],q[6];
ry(-1.60043882652068) q[6];
ry(-1.5712841716432895) q[7];
cx q[6],q[7];
ry(1.4606709672470728) q[6];
ry(0.8114405007664308) q[7];
cx q[6],q[7];
ry(-1.5676049302315447) q[7];
ry(2.0530862825553484) q[8];
cx q[7],q[8];
ry(3.1392012807957443) q[7];
ry(1.9736204596763651) q[8];
cx q[7],q[8];
ry(-1.0895367016013093) q[8];
ry(3.0771110067664873) q[9];
cx q[8],q[9];
ry(0.06649404327442662) q[8];
ry(1.5034278112637232) q[9];
cx q[8],q[9];
ry(0.06190441666980639) q[9];
ry(2.8988103920340245) q[10];
cx q[9],q[10];
ry(-1.5135621914046407) q[9];
ry(1.170198591122846) q[10];
cx q[9],q[10];
ry(-3.123666663758311) q[10];
ry(0.05062180918035471) q[11];
cx q[10],q[11];
ry(-3.0764703716445845) q[10];
ry(0.765869888820787) q[11];
cx q[10],q[11];
ry(-1.5550916064799196) q[11];
ry(2.6631601914113783) q[12];
cx q[11],q[12];
ry(0.0012587574368631384) q[11];
ry(-1.7634358139334887) q[12];
cx q[11],q[12];
ry(0.4831477726743645) q[12];
ry(-1.5834991079373755) q[13];
cx q[12],q[13];
ry(-0.30947112213229033) q[12];
ry(-1.5389013450426459) q[13];
cx q[12],q[13];
ry(-1.558391513518274) q[13];
ry(0.0880108594560518) q[14];
cx q[13],q[14];
ry(0.2347309732535782) q[13];
ry(-1.5760046979940325) q[14];
cx q[13],q[14];
ry(-2.967885972009459) q[14];
ry(1.5967236306083672) q[15];
cx q[14],q[15];
ry(1.5700580155382209) q[14];
ry(-2.4581987300555985) q[15];
cx q[14],q[15];
ry(-1.570814020292839) q[15];
ry(-1.575271362348427) q[16];
cx q[15],q[16];
ry(1.5712092057245117) q[15];
ry(1.7566993574447356) q[16];
cx q[15],q[16];
ry(1.5705195345096916) q[16];
ry(-2.394331883072987) q[17];
cx q[16],q[17];
ry(-1.5727644436520167) q[16];
ry(1.5590001360969905) q[17];
cx q[16],q[17];
ry(1.5708441384405116) q[17];
ry(-3.047180739186854) q[18];
cx q[17],q[18];
ry(1.5692044049666318) q[17];
ry(-1.5651668750750063) q[18];
cx q[17],q[18];
ry(1.5476156546804107) q[18];
ry(-2.5851929414994657) q[19];
cx q[18],q[19];
ry(1.5691611924738833) q[18];
ry(-0.1513848239736605) q[19];
cx q[18],q[19];
ry(-0.0768304390709007) q[0];
ry(1.4291549604527534) q[1];
ry(1.56720449442978) q[2];
ry(-1.426822016847361) q[3];
ry(-1.42642430295753) q[4];
ry(1.5687681287049913) q[5];
ry(-1.5767242566013886) q[6];
ry(-1.570761688533547) q[7];
ry(1.5697035890587534) q[8];
ry(-1.5705846480378702) q[9];
ry(-1.576339199315517) q[10];
ry(1.5704755169983657) q[11];
ry(-1.5719332104108035) q[12];
ry(-1.5718664924946788) q[13];
ry(-1.5699445526381277) q[14];
ry(-1.570615437686303) q[15];
ry(-1.5709760384662275) q[16];
ry(-1.5679562575171688) q[17];
ry(-1.5487486080265414) q[18];
ry(-1.5714450458977947) q[19];