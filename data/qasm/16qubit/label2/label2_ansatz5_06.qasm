OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(0.1851404125350351) q[0];
ry(-2.1395573341915997) q[1];
cx q[0],q[1];
ry(-1.9003158178456792) q[0];
ry(-0.21212718488235466) q[1];
cx q[0],q[1];
ry(0.012639706366315728) q[2];
ry(-2.7477286104143346) q[3];
cx q[2],q[3];
ry(-2.8837409422897746) q[2];
ry(-1.8428794690785972) q[3];
cx q[2],q[3];
ry(-2.593017811586102) q[4];
ry(-0.06713465502624594) q[5];
cx q[4],q[5];
ry(-3.0550625526372985) q[4];
ry(2.975506436023548) q[5];
cx q[4],q[5];
ry(-0.014332587618289757) q[6];
ry(2.0848459396261747) q[7];
cx q[6],q[7];
ry(-3.0229447996151735) q[6];
ry(2.030681760635143) q[7];
cx q[6],q[7];
ry(3.11674910415637) q[8];
ry(-0.07147725157898499) q[9];
cx q[8],q[9];
ry(1.5679146321504736) q[8];
ry(1.4766342250134201) q[9];
cx q[8],q[9];
ry(1.8968428312975467) q[10];
ry(-2.9769932556778786) q[11];
cx q[10],q[11];
ry(1.8110758211512217) q[10];
ry(1.2585225998109457) q[11];
cx q[10],q[11];
ry(-2.2450576726856433) q[12];
ry(-2.323155232909483) q[13];
cx q[12],q[13];
ry(0.11018225981435315) q[12];
ry(0.09302481649065175) q[13];
cx q[12],q[13];
ry(-0.09536616818293306) q[14];
ry(1.9280137587734476) q[15];
cx q[14],q[15];
ry(-2.6271602270159096) q[14];
ry(-0.44243045403813874) q[15];
cx q[14],q[15];
ry(-3.093479999567109) q[1];
ry(-2.861482688303929) q[2];
cx q[1],q[2];
ry(2.507954655943462) q[1];
ry(-1.4182953165480177) q[2];
cx q[1],q[2];
ry(1.6480947373353247) q[3];
ry(-1.0314855283431532) q[4];
cx q[3],q[4];
ry(-0.5637098964585814) q[3];
ry(-0.011990398651757616) q[4];
cx q[3],q[4];
ry(-1.2001635725749802) q[5];
ry(-1.7264041377270074) q[6];
cx q[5],q[6];
ry(-1.980346463477673) q[5];
ry(1.814941710641512) q[6];
cx q[5],q[6];
ry(0.14245959873849223) q[7];
ry(2.7468282402771558) q[8];
cx q[7],q[8];
ry(0.2402675125906841) q[7];
ry(0.03121101205318683) q[8];
cx q[7],q[8];
ry(-1.9155869132756036) q[9];
ry(2.9297442876692874) q[10];
cx q[9],q[10];
ry(0.24386134098985393) q[9];
ry(-0.16608442978275542) q[10];
cx q[9],q[10];
ry(-1.8979173906461109) q[11];
ry(1.3315433363395903) q[12];
cx q[11],q[12];
ry(3.1021675547406704) q[11];
ry(-0.07090947536734332) q[12];
cx q[11],q[12];
ry(-1.6353217047896595) q[13];
ry(-2.135389366560429) q[14];
cx q[13],q[14];
ry(-2.2288581690477205) q[13];
ry(-1.257751983497273) q[14];
cx q[13],q[14];
ry(1.2587974281134744) q[0];
ry(-0.1722643785175677) q[1];
cx q[0],q[1];
ry(2.813983590035687) q[0];
ry(-2.566168080194615) q[1];
cx q[0],q[1];
ry(2.2442555652360605) q[2];
ry(-2.600182476951309) q[3];
cx q[2],q[3];
ry(-0.01500382436812742) q[2];
ry(-3.107843519237021) q[3];
cx q[2],q[3];
ry(1.5497748102323028) q[4];
ry(-1.12303878718585) q[5];
cx q[4],q[5];
ry(0.005503584371021977) q[4];
ry(0.34833487125925633) q[5];
cx q[4],q[5];
ry(2.7327551370742342) q[6];
ry(-1.7232303599660475) q[7];
cx q[6],q[7];
ry(2.647733994605743) q[6];
ry(-0.8065513045505517) q[7];
cx q[6],q[7];
ry(0.5729934333682376) q[8];
ry(-3.1289987714699468) q[9];
cx q[8],q[9];
ry(0.2833382911247478) q[8];
ry(-0.6101338002331594) q[9];
cx q[8],q[9];
ry(1.2964583573630026) q[10];
ry(-2.1689372423470568) q[11];
cx q[10],q[11];
ry(-3.1083299532621496) q[10];
ry(-0.5283996654687142) q[11];
cx q[10],q[11];
ry(-1.6581804646305645) q[12];
ry(-2.905113661950841) q[13];
cx q[12],q[13];
ry(-0.4703026591746653) q[12];
ry(0.6783616884308971) q[13];
cx q[12],q[13];
ry(-0.1224961515182697) q[14];
ry(-3.055100303560828) q[15];
cx q[14],q[15];
ry(-2.795167224393558) q[14];
ry(-0.4090970512543146) q[15];
cx q[14],q[15];
ry(-1.2383977403118411) q[1];
ry(-0.4675777593143566) q[2];
cx q[1],q[2];
ry(-2.5268613920212495) q[1];
ry(-2.3139872088037676) q[2];
cx q[1],q[2];
ry(2.650570449470543) q[3];
ry(1.737427843238283) q[4];
cx q[3],q[4];
ry(-3.119235271603021) q[3];
ry(-0.009539696311176016) q[4];
cx q[3],q[4];
ry(0.6049134315896474) q[5];
ry(-2.8650861160693255) q[6];
cx q[5],q[6];
ry(-0.048421664935014874) q[5];
ry(-2.725068170039383) q[6];
cx q[5],q[6];
ry(-0.6919109017701909) q[7];
ry(-0.8465567369769209) q[8];
cx q[7],q[8];
ry(0.16737022442358587) q[7];
ry(3.117804742364046) q[8];
cx q[7],q[8];
ry(-1.1283182734478325) q[9];
ry(1.307558771235415) q[10];
cx q[9],q[10];
ry(-0.14055133754807336) q[9];
ry(-3.1114930372129437) q[10];
cx q[9],q[10];
ry(2.137212647431582) q[11];
ry(2.871112491224243) q[12];
cx q[11],q[12];
ry(-2.6070863689472796) q[11];
ry(0.06626530461640165) q[12];
cx q[11],q[12];
ry(-2.4134488982805045) q[13];
ry(1.6395619803881507) q[14];
cx q[13],q[14];
ry(-1.3778600681691353) q[13];
ry(-0.023066778334424498) q[14];
cx q[13],q[14];
ry(-2.5986981629407113) q[0];
ry(0.15438491259079257) q[1];
cx q[0],q[1];
ry(2.2798694458764057) q[0];
ry(0.29013080735401875) q[1];
cx q[0],q[1];
ry(-1.292951031994391) q[2];
ry(0.9279927239978312) q[3];
cx q[2],q[3];
ry(-2.8645668319740425) q[2];
ry(-0.09463347834137892) q[3];
cx q[2],q[3];
ry(-1.8289377331772136) q[4];
ry(-0.8620638259309329) q[5];
cx q[4],q[5];
ry(-2.22970755927786) q[4];
ry(2.5739186819493503) q[5];
cx q[4],q[5];
ry(2.580772623119403) q[6];
ry(2.1128023653710652) q[7];
cx q[6],q[7];
ry(2.2152364449811954) q[6];
ry(-0.9098975596401981) q[7];
cx q[6],q[7];
ry(-3.1403588637947695) q[8];
ry(2.4118947577837773) q[9];
cx q[8],q[9];
ry(-0.3835036051760543) q[8];
ry(-0.5889657976628833) q[9];
cx q[8],q[9];
ry(-0.5911675798484843) q[10];
ry(1.0197831065219296) q[11];
cx q[10],q[11];
ry(3.0709495406912746) q[10];
ry(-0.021012276574922842) q[11];
cx q[10],q[11];
ry(-1.2840348078496526) q[12];
ry(-0.05182972345850062) q[13];
cx q[12],q[13];
ry(-1.7572357470375382) q[12];
ry(-0.5299652005185616) q[13];
cx q[12],q[13];
ry(1.3616805268088727) q[14];
ry(2.6726220726882493) q[15];
cx q[14],q[15];
ry(-0.7557296185789868) q[14];
ry(0.5478305598024669) q[15];
cx q[14],q[15];
ry(-0.8318207148594421) q[1];
ry(2.623843631686367) q[2];
cx q[1],q[2];
ry(0.04248479384870649) q[1];
ry(3.02688461117873) q[2];
cx q[1],q[2];
ry(-1.4475237614625662) q[3];
ry(-1.4661722282627103) q[4];
cx q[3],q[4];
ry(-0.020039699700585164) q[3];
ry(1.0094787429565253) q[4];
cx q[3],q[4];
ry(1.1461267933538455) q[5];
ry(-0.3851862577692229) q[6];
cx q[5],q[6];
ry(-3.0116991390568377) q[5];
ry(2.8329910251433224) q[6];
cx q[5],q[6];
ry(-1.255037802014927) q[7];
ry(3.0397887138443176) q[8];
cx q[7],q[8];
ry(-2.171084775930723) q[7];
ry(-1.921662009736182) q[8];
cx q[7],q[8];
ry(-2.4884611912837507) q[9];
ry(0.23925936605744266) q[10];
cx q[9],q[10];
ry(-0.003349536573133527) q[9];
ry(-3.1012092550961374) q[10];
cx q[9],q[10];
ry(-2.2189656947788565) q[11];
ry(2.1700416961308187) q[12];
cx q[11],q[12];
ry(-1.7087299160272131) q[11];
ry(3.0888045468550502) q[12];
cx q[11],q[12];
ry(1.6393211987427048) q[13];
ry(-1.0302753074180417) q[14];
cx q[13],q[14];
ry(-0.05321926123570276) q[13];
ry(-0.08743123657220765) q[14];
cx q[13],q[14];
ry(0.4255779883420245) q[0];
ry(2.3875496482893954) q[1];
cx q[0],q[1];
ry(0.382970490173813) q[0];
ry(1.247796084298976) q[1];
cx q[0],q[1];
ry(1.5804591612598733) q[2];
ry(-0.15783754818441745) q[3];
cx q[2],q[3];
ry(-0.3651329528074832) q[2];
ry(-2.7523129787006346) q[3];
cx q[2],q[3];
ry(3.0483830780496968) q[4];
ry(-2.1031657391631224) q[5];
cx q[4],q[5];
ry(-2.9735692372601914) q[4];
ry(3.1300299964240383) q[5];
cx q[4],q[5];
ry(1.7445208131469387) q[6];
ry(-1.7713299203404569) q[7];
cx q[6],q[7];
ry(-0.31551254171953946) q[6];
ry(-0.12111058752661119) q[7];
cx q[6],q[7];
ry(0.40451031945347965) q[8];
ry(-0.19249818257760243) q[9];
cx q[8],q[9];
ry(-0.039888553805105254) q[8];
ry(-1.2933992480243852) q[9];
cx q[8],q[9];
ry(-3.0139893266250524) q[10];
ry(3.0343280424677714) q[11];
cx q[10],q[11];
ry(-1.2232995772017494) q[10];
ry(0.6618478138141618) q[11];
cx q[10],q[11];
ry(2.853716629354434) q[12];
ry(2.3704182035717096) q[13];
cx q[12],q[13];
ry(-0.4388431173958566) q[12];
ry(0.04447375978253909) q[13];
cx q[12],q[13];
ry(2.149585635109867) q[14];
ry(-0.6033895381238146) q[15];
cx q[14],q[15];
ry(-0.08243061668457538) q[14];
ry(0.9439073729976566) q[15];
cx q[14],q[15];
ry(1.2064060969481414) q[1];
ry(2.4863731571249073) q[2];
cx q[1],q[2];
ry(-0.968502373395618) q[1];
ry(0.1369313978318898) q[2];
cx q[1],q[2];
ry(-1.6960865260093794) q[3];
ry(-2.026374083493307) q[4];
cx q[3],q[4];
ry(-3.1358202498499734) q[3];
ry(0.1515976079520973) q[4];
cx q[3],q[4];
ry(-1.8356464751583306) q[5];
ry(2.833206894037761) q[6];
cx q[5],q[6];
ry(0.019131110662566442) q[5];
ry(-0.015477610539272211) q[6];
cx q[5],q[6];
ry(0.2827709393184438) q[7];
ry(-0.10551340692215572) q[8];
cx q[7],q[8];
ry(0.23435189017757724) q[7];
ry(1.2359329262386494) q[8];
cx q[7],q[8];
ry(-2.169097726509466) q[9];
ry(-0.7887231306535947) q[10];
cx q[9],q[10];
ry(-2.944500921659993) q[9];
ry(-0.005081213128518946) q[10];
cx q[9],q[10];
ry(-0.43808502889922174) q[11];
ry(0.24919918432374114) q[12];
cx q[11],q[12];
ry(-0.052747116759291224) q[11];
ry(-0.04798473878779656) q[12];
cx q[11],q[12];
ry(1.795870942008619) q[13];
ry(-1.8771967999768897) q[14];
cx q[13],q[14];
ry(-1.8221052633641284) q[13];
ry(-0.018667998963922905) q[14];
cx q[13],q[14];
ry(-0.3919770917166536) q[0];
ry(0.0012971871898841414) q[1];
cx q[0],q[1];
ry(-0.5899825192269592) q[0];
ry(-2.0477653467913655) q[1];
cx q[0],q[1];
ry(1.915453954583062) q[2];
ry(-1.7180274654786387) q[3];
cx q[2],q[3];
ry(2.5050657861297614) q[2];
ry(1.4894609148891576) q[3];
cx q[2],q[3];
ry(-0.2678090898307259) q[4];
ry(0.6717273788554401) q[5];
cx q[4],q[5];
ry(0.4717188591597541) q[4];
ry(-1.830313452003009) q[5];
cx q[4],q[5];
ry(-2.5639862086247147) q[6];
ry(1.3876744925388342) q[7];
cx q[6],q[7];
ry(-0.10637314215408723) q[6];
ry(-2.8408277713933057) q[7];
cx q[6],q[7];
ry(-2.944572382474914) q[8];
ry(-3.036313103823625) q[9];
cx q[8],q[9];
ry(-3.139603369534256) q[8];
ry(1.3357838925633583) q[9];
cx q[8],q[9];
ry(1.8697009888602427) q[10];
ry(-2.1981005332046246) q[11];
cx q[10],q[11];
ry(0.25290603097896286) q[10];
ry(0.43919470786589887) q[11];
cx q[10],q[11];
ry(-0.025557649626762213) q[12];
ry(1.517539668289312) q[13];
cx q[12],q[13];
ry(-1.7641572474902372) q[12];
ry(-1.0192498939545087) q[13];
cx q[12],q[13];
ry(-0.8534980285743881) q[14];
ry(-2.8965562997449457) q[15];
cx q[14],q[15];
ry(1.0085199237262517) q[14];
ry(-0.09103003476444606) q[15];
cx q[14],q[15];
ry(0.9318126110137068) q[1];
ry(-1.1870541397041432) q[2];
cx q[1],q[2];
ry(0.8161225705334688) q[1];
ry(1.8185280352685853) q[2];
cx q[1],q[2];
ry(0.37439512120854723) q[3];
ry(-2.1944285936400325) q[4];
cx q[3],q[4];
ry(-0.007530703055003961) q[3];
ry(-3.0493632088560925) q[4];
cx q[3],q[4];
ry(-1.9388364793956834) q[5];
ry(0.8575801304180375) q[6];
cx q[5],q[6];
ry(-0.12059467251096379) q[5];
ry(0.010458800388122934) q[6];
cx q[5],q[6];
ry(-0.4200806819932576) q[7];
ry(-3.0488022722849624) q[8];
cx q[7],q[8];
ry(0.27719215979089107) q[7];
ry(-2.617385106546454) q[8];
cx q[7],q[8];
ry(1.5874057442891625) q[9];
ry(-1.9683587556624413) q[10];
cx q[9],q[10];
ry(-2.922106238159778) q[9];
ry(-0.00025718376396492205) q[10];
cx q[9],q[10];
ry(1.5774536502007415) q[11];
ry(-1.7078995637606462) q[12];
cx q[11],q[12];
ry(-3.141166300040403) q[11];
ry(-0.03900278083008575) q[12];
cx q[11],q[12];
ry(0.8578762179223054) q[13];
ry(2.253682518991128) q[14];
cx q[13],q[14];
ry(2.8398900209671267) q[13];
ry(0.06827685839067676) q[14];
cx q[13],q[14];
ry(0.9694279459636652) q[0];
ry(-1.2511672191034662) q[1];
cx q[0],q[1];
ry(-0.46883126157867927) q[0];
ry(1.9515363634362868) q[1];
cx q[0],q[1];
ry(2.680968293134886) q[2];
ry(-1.5986647043466196) q[3];
cx q[2],q[3];
ry(0.9001087210653143) q[2];
ry(-0.003492677550739719) q[3];
cx q[2],q[3];
ry(1.671148271487473) q[4];
ry(-2.4503417292287053) q[5];
cx q[4],q[5];
ry(2.944163044354393) q[4];
ry(-1.5368224625459859) q[5];
cx q[4],q[5];
ry(-0.2394983699768658) q[6];
ry(2.1704031372782175) q[7];
cx q[6],q[7];
ry(-3.0808694921506743) q[6];
ry(3.124313743064644) q[7];
cx q[6],q[7];
ry(-2.542133977660026) q[8];
ry(1.8572408146015293) q[9];
cx q[8],q[9];
ry(1.2098559645873204) q[8];
ry(2.8252250468257176) q[9];
cx q[8],q[9];
ry(-1.7462007322167246) q[10];
ry(-2.2214196774091777) q[11];
cx q[10],q[11];
ry(0.9418528525923939) q[10];
ry(2.2505420006691947) q[11];
cx q[10],q[11];
ry(1.224502614934889) q[12];
ry(-2.7625554943023607) q[13];
cx q[12],q[13];
ry(0.07196136119317967) q[12];
ry(0.1306760953833912) q[13];
cx q[12],q[13];
ry(2.387351466629854) q[14];
ry(1.3676248976272234) q[15];
cx q[14],q[15];
ry(0.07583807185952234) q[14];
ry(0.7484638494895721) q[15];
cx q[14],q[15];
ry(-2.96036142157284) q[1];
ry(-2.7111739652535953) q[2];
cx q[1],q[2];
ry(0.32674297906845773) q[1];
ry(-1.5423942238668369) q[2];
cx q[1],q[2];
ry(1.8800467864457104) q[3];
ry(-2.3963532421589613) q[4];
cx q[3],q[4];
ry(-2.250840910327852) q[3];
ry(0.11315871394638834) q[4];
cx q[3],q[4];
ry(-0.710711084067106) q[5];
ry(0.13344947282461472) q[6];
cx q[5],q[6];
ry(-0.031514017504671976) q[5];
ry(3.1366412093397655) q[6];
cx q[5],q[6];
ry(-0.419111022562209) q[7];
ry(-0.8020936312455964) q[8];
cx q[7],q[8];
ry(-2.1363111850432412) q[7];
ry(2.642508505123322) q[8];
cx q[7],q[8];
ry(-1.0315373438461286) q[9];
ry(0.98691685381089) q[10];
cx q[9],q[10];
ry(-0.056699609867334716) q[9];
ry(-0.022412350191401174) q[10];
cx q[9],q[10];
ry(-1.3780605128935468) q[11];
ry(-1.2629236844952547) q[12];
cx q[11],q[12];
ry(1.857774150280009) q[11];
ry(2.275363517501778) q[12];
cx q[11],q[12];
ry(2.5852767541478605) q[13];
ry(2.8799690040847077) q[14];
cx q[13],q[14];
ry(-2.8793495216858096) q[13];
ry(-0.08748511032795707) q[14];
cx q[13],q[14];
ry(-1.4122951451770147) q[0];
ry(2.5199830154574183) q[1];
cx q[0],q[1];
ry(1.2461615672261814) q[0];
ry(-2.4362373717069405) q[1];
cx q[0],q[1];
ry(2.272965599646316) q[2];
ry(1.066297668527521) q[3];
cx q[2],q[3];
ry(3.1413359837277426) q[2];
ry(-0.18772000946302658) q[3];
cx q[2],q[3];
ry(1.569476284817418) q[4];
ry(-0.8431440124528775) q[5];
cx q[4],q[5];
ry(-3.1200135191364238) q[4];
ry(2.4675362360193116) q[5];
cx q[4],q[5];
ry(2.5910010025693264) q[6];
ry(-1.374544686674458) q[7];
cx q[6],q[7];
ry(3.108346608442271) q[6];
ry(2.82129581892415) q[7];
cx q[6],q[7];
ry(1.1208302361852356) q[8];
ry(0.29469884287643083) q[9];
cx q[8],q[9];
ry(-2.858128761291564) q[8];
ry(1.657108467551633) q[9];
cx q[8],q[9];
ry(2.983839689028935) q[10];
ry(-0.6443413238621203) q[11];
cx q[10],q[11];
ry(-0.002743772026017055) q[10];
ry(3.1308229705639072) q[11];
cx q[10],q[11];
ry(1.4882493815983642) q[12];
ry(1.650237989131427) q[13];
cx q[12],q[13];
ry(2.4151342791464847) q[12];
ry(3.0913418954427194) q[13];
cx q[12],q[13];
ry(0.2793060853817761) q[14];
ry(-0.4102138527286545) q[15];
cx q[14],q[15];
ry(0.6972194883108592) q[14];
ry(-0.6875267609963788) q[15];
cx q[14],q[15];
ry(-1.5268184258403998) q[1];
ry(-1.7686788219012985) q[2];
cx q[1],q[2];
ry(-2.4407983263704107) q[1];
ry(-0.18521962523314436) q[2];
cx q[1],q[2];
ry(1.734613613063443) q[3];
ry(1.5579778235416257) q[4];
cx q[3],q[4];
ry(-2.1729212777602287) q[3];
ry(1.5506877550390552) q[4];
cx q[3],q[4];
ry(-1.1327081813352577) q[5];
ry(0.9957045336785236) q[6];
cx q[5],q[6];
ry(-2.9952226187866433) q[5];
ry(-0.06613519730410289) q[6];
cx q[5],q[6];
ry(2.51201147906502) q[7];
ry(-0.5287464607140047) q[8];
cx q[7],q[8];
ry(1.045001992799687) q[7];
ry(1.5361832792799397) q[8];
cx q[7],q[8];
ry(-3.096028380352923) q[9];
ry(-2.902653281184454) q[10];
cx q[9],q[10];
ry(1.2818621048669) q[9];
ry(-3.1084387375839153) q[10];
cx q[9],q[10];
ry(-2.358391259736039) q[11];
ry(-1.7252858310020702) q[12];
cx q[11],q[12];
ry(0.9592581582407628) q[11];
ry(2.672067421905789) q[12];
cx q[11],q[12];
ry(-2.5000179836924086) q[13];
ry(0.8815605434893763) q[14];
cx q[13],q[14];
ry(3.107443779643313) q[13];
ry(0.010884548387086745) q[14];
cx q[13],q[14];
ry(-2.3074901596566835) q[0];
ry(2.8713240249188052) q[1];
cx q[0],q[1];
ry(-0.42278357141144857) q[0];
ry(2.065729683062463) q[1];
cx q[0],q[1];
ry(-2.569971356771104) q[2];
ry(2.64434036484999) q[3];
cx q[2],q[3];
ry(-3.125292933602033) q[2];
ry(3.1211278108881064) q[3];
cx q[2],q[3];
ry(-1.6667752490203998) q[4];
ry(0.060532336803603044) q[5];
cx q[4],q[5];
ry(-0.08206891527646376) q[4];
ry(-3.1403569140040317) q[5];
cx q[4],q[5];
ry(-2.1697969555236645) q[6];
ry(-2.2623833463318794) q[7];
cx q[6],q[7];
ry(3.127338257407085) q[6];
ry(-1.4473608063553818) q[7];
cx q[6],q[7];
ry(-0.5147690447290874) q[8];
ry(-0.39637466751884265) q[9];
cx q[8],q[9];
ry(0.03288178679015186) q[8];
ry(-2.508650219855189) q[9];
cx q[8],q[9];
ry(-0.6006255908441123) q[10];
ry(-1.8302762913782256) q[11];
cx q[10],q[11];
ry(0.0021775890298867976) q[10];
ry(-3.1388714452041455) q[11];
cx q[10],q[11];
ry(1.6558626379641816) q[12];
ry(2.252265913607091) q[13];
cx q[12],q[13];
ry(-0.14263994709657088) q[12];
ry(-0.07556143894489244) q[13];
cx q[12],q[13];
ry(2.028061008000717) q[14];
ry(0.1437213773285964) q[15];
cx q[14],q[15];
ry(2.5631791666993697) q[14];
ry(-2.722187468583209) q[15];
cx q[14],q[15];
ry(-1.7308785709236498) q[1];
ry(1.4252803939498389) q[2];
cx q[1],q[2];
ry(0.43197903673021837) q[1];
ry(0.39919388635795267) q[2];
cx q[1],q[2];
ry(2.6900932093642917) q[3];
ry(3.049891342322029) q[4];
cx q[3],q[4];
ry(-3.109486169905438) q[3];
ry(-1.4874673279866597) q[4];
cx q[3],q[4];
ry(1.0508194339378876) q[5];
ry(-2.791671760236334) q[6];
cx q[5],q[6];
ry(3.0566047969789114) q[5];
ry(3.072293292207773) q[6];
cx q[5],q[6];
ry(-0.0592707961145308) q[7];
ry(-0.14673332660808547) q[8];
cx q[7],q[8];
ry(1.3280186763762236) q[7];
ry(-3.0938585949440105) q[8];
cx q[7],q[8];
ry(2.241476830142953) q[9];
ry(2.465126566904036) q[10];
cx q[9],q[10];
ry(-1.2605419081767266) q[9];
ry(-3.096448220466464) q[10];
cx q[9],q[10];
ry(1.8713223527951799) q[11];
ry(-0.15565355736466202) q[12];
cx q[11],q[12];
ry(-0.20661025371502983) q[11];
ry(0.4413121593240499) q[12];
cx q[11],q[12];
ry(-0.20927294167694185) q[13];
ry(-1.870181094743459) q[14];
cx q[13],q[14];
ry(-0.05073024489311671) q[13];
ry(3.0507971764310278) q[14];
cx q[13],q[14];
ry(2.7097935309865244) q[0];
ry(-1.8311446464399481) q[1];
cx q[0],q[1];
ry(1.1353905773151949) q[0];
ry(2.802845291116507) q[1];
cx q[0],q[1];
ry(1.2658683931016477) q[2];
ry(-2.138681879707891) q[3];
cx q[2],q[3];
ry(0.8111833507991745) q[2];
ry(-1.5042271939818777) q[3];
cx q[2],q[3];
ry(1.160213347242594) q[4];
ry(-0.10959623242513614) q[5];
cx q[4],q[5];
ry(-3.133802435211502) q[4];
ry(3.08397223457016) q[5];
cx q[4],q[5];
ry(2.3108346791322854) q[6];
ry(-1.595356023731238) q[7];
cx q[6],q[7];
ry(-2.761388114810849) q[6];
ry(0.830556666114114) q[7];
cx q[6],q[7];
ry(-1.0228941603188364) q[8];
ry(2.1013361806192563) q[9];
cx q[8],q[9];
ry(-1.0311552931370045) q[8];
ry(1.9461974348140536) q[9];
cx q[8],q[9];
ry(-3.0151938599638903) q[10];
ry(0.021685948936294754) q[11];
cx q[10],q[11];
ry(3.1377773273626337) q[10];
ry(0.13982428652507473) q[11];
cx q[10],q[11];
ry(-1.1379750497569132) q[12];
ry(0.49576118021804305) q[13];
cx q[12],q[13];
ry(0.10179991796093034) q[12];
ry(-3.0133287359749987) q[13];
cx q[12],q[13];
ry(1.4297334999721398) q[14];
ry(0.23005909027974436) q[15];
cx q[14],q[15];
ry(-2.3364648260142267) q[14];
ry(-2.497735699676555) q[15];
cx q[14],q[15];
ry(-0.48877216025891407) q[1];
ry(-1.889130014835543) q[2];
cx q[1],q[2];
ry(-0.03987327608750024) q[1];
ry(0.08576635897934126) q[2];
cx q[1],q[2];
ry(-0.44718311021575374) q[3];
ry(0.09230044192912025) q[4];
cx q[3],q[4];
ry(0.005676095660254326) q[3];
ry(-3.095579535669052) q[4];
cx q[3],q[4];
ry(2.7334392716350706) q[5];
ry(1.8717506182444668) q[6];
cx q[5],q[6];
ry(-0.10489929087241645) q[5];
ry(-0.05282011498360096) q[6];
cx q[5],q[6];
ry(1.4801865967644339) q[7];
ry(-0.12082856447260255) q[8];
cx q[7],q[8];
ry(-3.0146926356335473) q[7];
ry(0.021614041052055555) q[8];
cx q[7],q[8];
ry(-1.1821446804723914) q[9];
ry(-1.1002734994745884) q[10];
cx q[9],q[10];
ry(0.0313405527790715) q[9];
ry(0.009151392549018455) q[10];
cx q[9],q[10];
ry(1.9440586346743398) q[11];
ry(2.0113926008578575) q[12];
cx q[11],q[12];
ry(-0.8961950223707237) q[11];
ry(3.0453066474867376) q[12];
cx q[11],q[12];
ry(-0.4834064089260588) q[13];
ry(-2.7157315492312986) q[14];
cx q[13],q[14];
ry(-3.1376166336881597) q[13];
ry(-3.0764068807722995) q[14];
cx q[13],q[14];
ry(-2.0259658120114876) q[0];
ry(-1.308329112257275) q[1];
ry(-3.125163080216845) q[2];
ry(-0.6147005998036796) q[3];
ry(-1.4097061795945935) q[4];
ry(1.0937326228246003) q[5];
ry(-1.5022857259309443) q[6];
ry(0.17096784246511998) q[7];
ry(1.5030519329938148) q[8];
ry(-1.6970822376605115) q[9];
ry(-0.46681403575279246) q[10];
ry(-1.0558114160325338) q[11];
ry(3.0418142173966367) q[12];
ry(-0.6023557304471009) q[13];
ry(-0.8181036312066801) q[14];
ry(-2.549086438632491) q[15];