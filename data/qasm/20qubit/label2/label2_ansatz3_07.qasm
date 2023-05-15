OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.5722243175483859) q[0];
rz(-1.2160733931827903) q[0];
ry(1.5730097153289468) q[1];
rz(-2.950216345866642) q[1];
ry(-0.33324102539611844) q[2];
rz(3.04118842902413) q[2];
ry(-3.105982891196012) q[3];
rz(0.8858074159313576) q[3];
ry(3.140924359575159) q[4];
rz(-0.7766241406685941) q[4];
ry(0.5049076027602448) q[5];
rz(2.26236169520003) q[5];
ry(-0.04042659490589249) q[6];
rz(-2.639855948044809) q[6];
ry(-3.1359879201034486) q[7];
rz(2.738867796057692) q[7];
ry(-1.847096025757832) q[8];
rz(-2.8154204557795106) q[8];
ry(-0.18695916074286426) q[9];
rz(2.4399262116119105) q[9];
ry(-0.0025613749808863773) q[10];
rz(-1.118337963740236) q[10];
ry(-1.5141276671835768) q[11];
rz(1.6053448084192254) q[11];
ry(-0.12398239229577938) q[12];
rz(-0.4380820396658151) q[12];
ry(-0.05955608580613779) q[13];
rz(-1.4300582060549896) q[13];
ry(0.05699728859133953) q[14];
rz(-0.5603397067738567) q[14];
ry(-3.1113124512112957) q[15];
rz(-0.37779161683699236) q[15];
ry(-0.002655631712807452) q[16];
rz(-2.0701531614177213) q[16];
ry(-3.1135599007760346) q[17];
rz(-2.8673094217599733) q[17];
ry(-0.5194147612676128) q[18];
rz(1.8840535571647603) q[18];
ry(-2.8991885345790815) q[19];
rz(-1.0685281609608763) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-2.2882427626259423) q[0];
rz(2.201557404688979) q[0];
ry(0.8620975614344304) q[1];
rz(1.9100331164177824) q[1];
ry(-1.581685902152764) q[2];
rz(1.4730642926946855) q[2];
ry(-3.1336583228877157) q[3];
rz(2.0876319289016596) q[3];
ry(-3.1406683713450505) q[4];
rz(-2.3852468179053985) q[4];
ry(2.6928353806673795) q[5];
rz(1.467460152011693) q[5];
ry(-1.5592719098369887) q[6];
rz(-2.5362779898406096) q[6];
ry(-3.13358848508171) q[7];
rz(2.4279291123342626) q[7];
ry(-2.7902899524251756) q[8];
rz(-0.681906303004737) q[8];
ry(3.139202008893137) q[9];
rz(-0.6553757709669181) q[9];
ry(-1.5837505360548219) q[10];
rz(-0.16271129592079075) q[10];
ry(0.9636632670890917) q[11];
rz(-0.8134690808316574) q[11];
ry(-1.455090106572734) q[12];
rz(-0.1281184497188756) q[12];
ry(0.0028904505795352975) q[13];
rz(2.2995809900561626) q[13];
ry(-3.0868296101563906) q[14];
rz(1.353606523187521) q[14];
ry(-2.913084529200411) q[15];
rz(2.2722842137108366) q[15];
ry(3.1407341987753563) q[16];
rz(1.9771232312263374) q[16];
ry(0.008429318690561516) q[17];
rz(-0.6324115864824104) q[17];
ry(-0.9724298228076171) q[18];
rz(-2.990552821273717) q[18];
ry(-1.8265327419885653) q[19];
rz(-0.8510389716993519) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(1.4288530416674028) q[0];
rz(2.354816185232243) q[0];
ry(3.1389221490112855) q[1];
rz(-1.3056189659895374) q[1];
ry(-0.7422908842084945) q[2];
rz(2.3732576345771585) q[2];
ry(-1.570420419528425) q[3];
rz(1.6684606865875962) q[3];
ry(-1.6862419075579544) q[4];
rz(-0.5146754236657296) q[4];
ry(-2.117765851980448) q[5];
rz(1.9138156778659343) q[5];
ry(2.8970631060090195) q[6];
rz(-2.425413532402446) q[6];
ry(-3.1363350746140815) q[7];
rz(1.9961022720346386) q[7];
ry(-1.5539686133942567) q[8];
rz(-0.7191433356863652) q[8];
ry(-0.32784935852716746) q[9];
rz(0.4461076786241053) q[9];
ry(3.1401873688478683) q[10];
rz(2.447796655882832) q[10];
ry(-0.6056869599903081) q[11];
rz(-1.2027777541939344) q[11];
ry(0.9175391330704823) q[12];
rz(-2.847277143895605) q[12];
ry(-1.5807754280665796) q[13];
rz(-2.9840580433598025) q[13];
ry(1.572218938107033) q[14];
rz(3.0498337788215846) q[14];
ry(-2.6439672332440236) q[15];
rz(2.3517229917766294) q[15];
ry(0.008610221129948189) q[16];
rz(-0.14108527544025673) q[16];
ry(-0.121241263355664) q[17];
rz(-1.8966805839560514) q[17];
ry(-0.23887778023101625) q[18];
rz(2.2083457263531026) q[18];
ry(0.37581253019282757) q[19];
rz(-3.1115406263397274) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(1.2307485327730632) q[0];
rz(2.9269336158848107) q[0];
ry(-1.5262278642319627) q[1];
rz(2.196935818955974) q[1];
ry(-0.06840848259823762) q[2];
rz(-3.087245385563779) q[2];
ry(0.03613985387697559) q[3];
rz(-2.6099201849638813) q[3];
ry(3.140861352027001) q[4];
rz(2.5753293422161287) q[4];
ry(1.6933298243186308) q[5];
rz(-3.0493700157699717) q[5];
ry(3.1372567086462073) q[6];
rz(0.45986993117321173) q[6];
ry(0.003725300370434503) q[7];
rz(-1.0454004722783437) q[7];
ry(-2.9620081935233795) q[8];
rz(-2.19153299890198) q[8];
ry(-0.0013919540085978034) q[9];
rz(0.2577252943598438) q[9];
ry(3.1383722559535707) q[10];
rz(-1.4738153446960203) q[10];
ry(-3.140340952582604) q[11];
rz(0.6447593919138749) q[11];
ry(1.5727536628259626) q[12];
rz(-1.572874674703729) q[12];
ry(3.141465406620365) q[13];
rz(0.1592348276039469) q[13];
ry(0.0033967340815056796) q[14];
rz(0.08733425980015072) q[14];
ry(-0.0025397240390189523) q[15];
rz(-0.8538650803790172) q[15];
ry(3.1396305666282656) q[16];
rz(-1.9885168044645536) q[16];
ry(3.1250167039973977) q[17];
rz(2.347690397173537) q[17];
ry(2.6456745696218067) q[18];
rz(3.0053076112431762) q[18];
ry(2.3882728062826404) q[19];
rz(1.6540059521500068) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(3.1236634917773207) q[0];
rz(-1.380978782977589) q[0];
ry(3.1102768014643694) q[1];
rz(0.6829640029805266) q[1];
ry(2.004606291719737) q[2];
rz(0.11116446101423881) q[2];
ry(3.1387710002403977) q[3];
rz(0.8957207330396502) q[3];
ry(2.832341257060569) q[4];
rz(-3.0544146841167716) q[4];
ry(1.7393453769454572) q[5];
rz(-1.7358102381918168) q[5];
ry(-0.19626944117598202) q[6];
rz(0.7766811809184996) q[6];
ry(3.1306541666988816) q[7];
rz(-3.1286845088288517) q[7];
ry(0.31928110174724905) q[8];
rz(2.27892867012571) q[8];
ry(0.29456554844759175) q[9];
rz(0.8330413381841234) q[9];
ry(-0.00014786043892467404) q[10];
rz(-3.1296071316908725) q[10];
ry(-1.3456713608060458) q[11];
rz(0.2935788658142142) q[11];
ry(1.570468463634393) q[12];
rz(-2.560086911988383) q[12];
ry(-1.537190335442439) q[13];
rz(2.7106599555836053) q[13];
ry(-1.625701173607644) q[14];
rz(-1.5391850991500222) q[14];
ry(1.3991811285343105) q[15];
rz(-0.36019486940088097) q[15];
ry(1.575669667974026) q[16];
rz(1.584049172815429) q[16];
ry(1.4767785650236789) q[17];
rz(-3.0356846730673372) q[17];
ry(1.2716841925897473) q[18];
rz(-2.9072158648196997) q[18];
ry(0.4043141799961432) q[19];
rz(-1.4435547762402843) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-0.7730035835795661) q[0];
rz(2.193740113744727) q[0];
ry(-0.6738117225242765) q[1];
rz(1.6078294337925367) q[1];
ry(-3.0739527337827575) q[2];
rz(1.2171578620197097) q[2];
ry(3.1365576523409366) q[3];
rz(2.0039810404497604) q[3];
ry(-0.0018640520925196926) q[4];
rz(2.3803778589188767) q[4];
ry(-2.544094092865083) q[5];
rz(2.5267481745373077) q[5];
ry(-3.136276246808237) q[6];
rz(-0.5139591135639527) q[6];
ry(-3.1355358006129483) q[7];
rz(2.040587593981388) q[7];
ry(-1.526278945759955) q[8];
rz(2.8169120592168757) q[8];
ry(-0.007077271950277186) q[9];
rz(-2.7012852000268825) q[9];
ry(-1.871984082153742) q[10];
rz(-0.25085890530244936) q[10];
ry(-0.04546664426001801) q[11];
rz(-3.0130870137735863) q[11];
ry(-0.9770073940885075) q[12];
rz(0.5504854688233232) q[12];
ry(-3.140808138917483) q[13];
rz(-3.0765498831409346) q[13];
ry(1.563742547428851) q[14];
rz(-0.4520397467675928) q[14];
ry(1.5248478501357106) q[15];
rz(-1.4494702913743396) q[15];
ry(-1.5621353466114236) q[16];
rz(1.5610713117934463) q[16];
ry(-3.1403456978596083) q[17];
rz(0.03331271963092419) q[17];
ry(-3.1334454579200677) q[18];
rz(-1.8605774033822442) q[18];
ry(-3.1191140834066187) q[19];
rz(-1.9499796999759904) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.2905501537895379) q[0];
rz(0.10803312102198308) q[0];
ry(0.06628699488585557) q[1];
rz(2.7235845646491055) q[1];
ry(2.6578692561231163) q[2];
rz(0.5599573861520208) q[2];
ry(1.5729820243498036) q[3];
rz(0.8798475658285982) q[3];
ry(-2.0048588075184153) q[4];
rz(-1.504002738239806) q[4];
ry(-0.013246836243716587) q[5];
rz(-1.7429543057680679) q[5];
ry(-2.581178854083794) q[6];
rz(1.3810588701705735) q[6];
ry(0.031454819723472166) q[7];
rz(0.7044791072488614) q[7];
ry(-3.1257272963675384) q[8];
rz(-1.8069079429440345) q[8];
ry(1.5975981758416062) q[9];
rz(-1.4461928454643764) q[9];
ry(3.1388957985567996) q[10];
rz(-0.250210390962466) q[10];
ry(-3.1279344516020813) q[11];
rz(2.899194284052558) q[11];
ry(0.0028052385681949232) q[12];
rz(2.750450652888661) q[12];
ry(3.140159853883596) q[13];
rz(2.0598403221150896) q[13];
ry(0.0034131921177902314) q[14];
rz(0.4475106380349651) q[14];
ry(-1.5782772846152466) q[15];
rz(1.5633869565725966) q[15];
ry(1.570632254514062) q[16];
rz(-1.0967567157097553) q[16];
ry(-1.5794338972744173) q[17];
rz(2.2520599491804365) q[17];
ry(0.0370699299256339) q[18];
rz(-0.28617794110511996) q[18];
ry(-1.6047970238690175) q[19];
rz(2.4762248057619285) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.23847612596204826) q[0];
rz(-2.1216984701596115) q[0];
ry(1.5813093588497957) q[1];
rz(2.313269259094074) q[1];
ry(-3.115643868081682) q[2];
rz(1.5187570624253366) q[2];
ry(-0.05100523676303523) q[3];
rz(-2.4989419981588004) q[3];
ry(3.140723280040032) q[4];
rz(1.6343904048921356) q[4];
ry(0.0058275821721819245) q[5];
rz(0.5354696789651845) q[5];
ry(3.1404194947966224) q[6];
rz(-1.7307788791891012) q[6];
ry(3.1413060025802495) q[7];
rz(-1.854314649868845) q[7];
ry(1.5873715386359935) q[8];
rz(-2.9834404015465) q[8];
ry(0.053411883953161876) q[9];
rz(-0.08561206835021994) q[9];
ry(-1.2542196497094518) q[10];
rz(3.1386473724462314) q[10];
ry(-0.001540406737127853) q[11];
rz(-2.9719290050792404) q[11];
ry(-2.5403160425754) q[12];
rz(1.7769065023735804) q[12];
ry(3.135403299122278) q[13];
rz(-2.279438032068756) q[13];
ry(-1.5728387018944137) q[14];
rz(-3.13867722954648) q[14];
ry(-1.5718283054738134) q[15];
rz(-2.1143197162719325) q[15];
ry(-3.1315547027880015) q[16];
rz(-2.7299048017338383) q[16];
ry(3.126147698943079) q[17];
rz(2.7144313896809336) q[17];
ry(1.5907623462808136) q[18];
rz(-0.3588223760458271) q[18];
ry(0.15046445332249905) q[19];
rz(0.9297978801702106) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-3.0557329816070187) q[0];
rz(2.535644385048452) q[0];
ry(3.131760834612426) q[1];
rz(0.25779832339124464) q[1];
ry(-1.2577523321587236) q[2];
rz(1.7952814421372667) q[2];
ry(-1.5694100232558617) q[3];
rz(-1.617042642372817) q[3];
ry(1.4683218981677342) q[4];
rz(0.9063190203554861) q[4];
ry(-1.6806684589767513) q[5];
rz(0.05205799367833797) q[5];
ry(1.556252349961479) q[6];
rz(-0.7368218004863004) q[6];
ry(3.1075090535217518) q[7];
rz(2.9075356195858695) q[7];
ry(-2.9607613393240437) q[8];
rz(-3.0619544214293377) q[8];
ry(-1.5319117730667033) q[9];
rz(-1.8107801239106536) q[9];
ry(3.133434796506437) q[10];
rz(-0.11989406570364469) q[10];
ry(0.9022147882138569) q[11];
rz(-1.4794543123603265) q[11];
ry(-2.8879841445199705) q[12];
rz(-2.747097512776223) q[12];
ry(-0.34372159434570815) q[13];
rz(-1.2850928651075213) q[13];
ry(1.5819792002590933) q[14];
rz(-2.1002549821040475) q[14];
ry(0.8150547751794763) q[15];
rz(-2.243614207346135) q[15];
ry(2.1571379582001304) q[16];
rz(-0.00019395289054888014) q[16];
ry(-3.089591255628301) q[17];
rz(-0.1600749647574799) q[17];
ry(0.02259264892767763) q[18];
rz(1.3790735269999432) q[18];
ry(2.796907859946289) q[19];
rz(0.6730203375113073) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.059050318371123955) q[0];
rz(3.1396465650589667) q[0];
ry(0.029565878573701635) q[1];
rz(0.8768053327705401) q[1];
ry(-3.1383919944866334) q[2];
rz(1.3290126295003981) q[2];
ry(1.5701359928573275) q[3];
rz(0.2931295211668381) q[3];
ry(3.1386932551038713) q[4];
rz(-0.10073182372833767) q[4];
ry(1.553674201314996) q[5];
rz(0.12161097844399649) q[5];
ry(0.006582490115175865) q[6];
rz(-1.0557220912908283) q[6];
ry(0.0001698026886476356) q[7];
rz(-2.0483873133071837) q[7];
ry(-1.5696630862444665) q[8];
rz(2.8595804829245277) q[8];
ry(0.021701451905182367) q[9];
rz(2.6830774763924823) q[9];
ry(-3.1184784523100553) q[10];
rz(-2.755084727523318) q[10];
ry(-0.006351026143929239) q[11];
rz(1.4759674906590665) q[11];
ry(-0.00035320777232428213) q[12];
rz(-2.5261588804368) q[12];
ry(3.139848775464765) q[13];
rz(1.153058058298968) q[13];
ry(0.0043098487125728495) q[14];
rz(1.8413924890514486) q[14];
ry(3.1354099010409957) q[15];
rz(0.2701338005927739) q[15];
ry(1.5655834672250337) q[16];
rz(-2.0400225053136345) q[16];
ry(0.010374275545638828) q[17];
rz(1.231891340708232) q[17];
ry(-3.054298055945608) q[18];
rz(-1.8893012860132607) q[18];
ry(0.1822777124370497) q[19];
rz(0.08958448286039543) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.43117612976582) q[0];
rz(-1.4110541118866369) q[0];
ry(2.9618455743411327) q[1];
rz(0.49756976941952663) q[1];
ry(-1.5171917130524186) q[2];
rz(1.7067850414910324) q[2];
ry(0.09632336644166717) q[3];
rz(2.6790820842756826) q[3];
ry(2.8261327569165107) q[4];
rz(2.3439280729872296) q[4];
ry(1.8379270454142205) q[5];
rz(-0.38581546864346183) q[5];
ry(1.6816476210237115) q[6];
rz(0.5969804331947558) q[6];
ry(0.2507298894493885) q[7];
rz(1.7866229726441984) q[7];
ry(1.9962484249951995) q[8];
rz(-2.434145449534587) q[8];
ry(1.8828268223914946) q[9];
rz(0.5115863195996649) q[9];
ry(2.3533041834666526) q[10];
rz(-2.5131939659535214) q[10];
ry(0.45963872266937855) q[11];
rz(1.4664393171828405) q[11];
ry(-0.06870867099956168) q[12];
rz(2.5483006936925734) q[12];
ry(0.20044337692030556) q[13];
rz(1.9557559962953328) q[13];
ry(0.46059133879721514) q[14];
rz(2.6413163249159455) q[14];
ry(2.320243149234605) q[15];
rz(0.8721974755791857) q[15];
ry(2.7945778492431406) q[16];
rz(-0.8466760864765286) q[16];
ry(2.2724870351504816) q[17];
rz(0.05144031524117723) q[17];
ry(-1.5561753396602085) q[18];
rz(-1.096450198792166) q[18];
ry(2.983476685262388) q[19];
rz(1.619261013945551) q[19];