OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.2349301181822345) q[0];
ry(2.4436722684289696) q[1];
cx q[0],q[1];
ry(-0.6512128076707242) q[0];
ry(-1.0643873874255645) q[1];
cx q[0],q[1];
ry(-2.882411958183161) q[2];
ry(-0.9406576392634749) q[3];
cx q[2],q[3];
ry(-2.5292015613603396) q[2];
ry(0.8458558697084948) q[3];
cx q[2],q[3];
ry(-2.0895569488167784) q[4];
ry(-1.481286701477295) q[5];
cx q[4],q[5];
ry(1.409627270689132) q[4];
ry(1.4358221971336704) q[5];
cx q[4],q[5];
ry(-1.9904135904381182) q[6];
ry(0.8699125092166139) q[7];
cx q[6],q[7];
ry(2.935314218836433) q[6];
ry(2.6104828957141177) q[7];
cx q[6],q[7];
ry(1.075713243807839) q[0];
ry(-1.4865848884787138) q[2];
cx q[0],q[2];
ry(-0.7533931122541857) q[0];
ry(-2.291523785866411) q[2];
cx q[0],q[2];
ry(-0.2564897506819174) q[2];
ry(1.5313042534377637) q[4];
cx q[2],q[4];
ry(2.247676060443296) q[2];
ry(1.1511374069228042) q[4];
cx q[2],q[4];
ry(-2.8246836405591096) q[4];
ry(0.6759077909779253) q[6];
cx q[4],q[6];
ry(1.5630211695480283) q[4];
ry(1.3593028542533814) q[6];
cx q[4],q[6];
ry(0.10647945742787888) q[1];
ry(-2.6641099295593014) q[3];
cx q[1],q[3];
ry(-1.7310281129775644) q[1];
ry(0.25687731963988725) q[3];
cx q[1],q[3];
ry(2.1212596682245106) q[3];
ry(1.7474901346675005) q[5];
cx q[3],q[5];
ry(0.39923880885182816) q[3];
ry(2.247070845964248) q[5];
cx q[3],q[5];
ry(0.8286923747964661) q[5];
ry(1.82656732394263) q[7];
cx q[5],q[7];
ry(1.2710683358239576) q[5];
ry(-1.8684984221496785) q[7];
cx q[5],q[7];
ry(2.969822500282573) q[0];
ry(-2.960085673448625) q[3];
cx q[0],q[3];
ry(-0.3475957620596354) q[0];
ry(-1.0403593777131297) q[3];
cx q[0],q[3];
ry(0.9861596731101452) q[1];
ry(-1.7748529763971925) q[2];
cx q[1],q[2];
ry(-0.24865959104556534) q[1];
ry(-1.9231236831103944) q[2];
cx q[1],q[2];
ry(-1.0131712809108082) q[2];
ry(-2.26583245286057) q[5];
cx q[2],q[5];
ry(1.405913059944602) q[2];
ry(-1.5340443978547929) q[5];
cx q[2],q[5];
ry(-2.230497752958703) q[3];
ry(2.8801092272890276) q[4];
cx q[3],q[4];
ry(-2.894176911080486) q[3];
ry(1.075332724580808) q[4];
cx q[3],q[4];
ry(-1.9082254366544475) q[4];
ry(-2.545992188771153) q[7];
cx q[4],q[7];
ry(0.6080980630116528) q[4];
ry(1.8089306218261043) q[7];
cx q[4],q[7];
ry(1.7628288758778738) q[5];
ry(-2.963775959767613) q[6];
cx q[5],q[6];
ry(-2.081379488541417) q[5];
ry(-1.8224885726312572) q[6];
cx q[5],q[6];
ry(-1.1332599567704098) q[0];
ry(1.5805370527824367) q[1];
cx q[0],q[1];
ry(-0.03325617260280733) q[0];
ry(-0.5537968304495449) q[1];
cx q[0],q[1];
ry(1.497221835251324) q[2];
ry(-1.73855438721876) q[3];
cx q[2],q[3];
ry(-0.535880723463964) q[2];
ry(0.9788297126419714) q[3];
cx q[2],q[3];
ry(0.481510431036412) q[4];
ry(2.2242198880450617) q[5];
cx q[4],q[5];
ry(2.2774357973843324) q[4];
ry(-0.9113912991215175) q[5];
cx q[4],q[5];
ry(-2.041259538526867) q[6];
ry(0.9652222699046158) q[7];
cx q[6],q[7];
ry(1.2991264461439975) q[6];
ry(-1.5994015038480134) q[7];
cx q[6],q[7];
ry(-2.5575464057637736) q[0];
ry(-0.05556823200323067) q[2];
cx q[0],q[2];
ry(0.17358051693350163) q[0];
ry(1.247515045114854) q[2];
cx q[0],q[2];
ry(-1.1981458724039198) q[2];
ry(1.1787005544816775) q[4];
cx q[2],q[4];
ry(-1.306317073245027) q[2];
ry(0.103091315458812) q[4];
cx q[2],q[4];
ry(-3.061636839281022) q[4];
ry(-1.286919177985012) q[6];
cx q[4],q[6];
ry(-2.359324060282304) q[4];
ry(1.8618717732977181) q[6];
cx q[4],q[6];
ry(1.8497881039300832) q[1];
ry(1.1400000489941697) q[3];
cx q[1],q[3];
ry(2.0006990926082797) q[1];
ry(-2.6924650820819296) q[3];
cx q[1],q[3];
ry(-2.2330655501232357) q[3];
ry(0.2076168901842452) q[5];
cx q[3],q[5];
ry(2.158738389701025) q[3];
ry(-0.015777902956528145) q[5];
cx q[3],q[5];
ry(-2.4819081914790737) q[5];
ry(1.679096113986688) q[7];
cx q[5],q[7];
ry(-0.5961135192862637) q[5];
ry(2.3761696678012174) q[7];
cx q[5],q[7];
ry(2.250962338230071) q[0];
ry(0.7127225031516975) q[3];
cx q[0],q[3];
ry(-1.4276497752830766) q[0];
ry(0.27088547340855396) q[3];
cx q[0],q[3];
ry(-0.33555161182923765) q[1];
ry(1.1956953884797041) q[2];
cx q[1],q[2];
ry(0.6662073892266216) q[1];
ry(-2.9820925218424996) q[2];
cx q[1],q[2];
ry(2.5839853774548707) q[2];
ry(-2.9009388390717286) q[5];
cx q[2],q[5];
ry(0.955886771966679) q[2];
ry(0.7054839119914007) q[5];
cx q[2],q[5];
ry(-0.5093281093640385) q[3];
ry(-0.23379307806158864) q[4];
cx q[3],q[4];
ry(1.804522191461824) q[3];
ry(0.937506085543915) q[4];
cx q[3],q[4];
ry(2.116352781140458) q[4];
ry(0.6161509692403195) q[7];
cx q[4],q[7];
ry(-0.3077148373946031) q[4];
ry(-1.0873600929989096) q[7];
cx q[4],q[7];
ry(2.8514481414648802) q[5];
ry(-2.250635558156106) q[6];
cx q[5],q[6];
ry(-1.6584502262459466) q[5];
ry(0.911843049054339) q[6];
cx q[5],q[6];
ry(-1.3858533572265115) q[0];
ry(-0.6405732465107974) q[1];
cx q[0],q[1];
ry(-2.4073202678013175) q[0];
ry(3.13509305566697) q[1];
cx q[0],q[1];
ry(-0.5170739498896821) q[2];
ry(-0.9193787483440694) q[3];
cx q[2],q[3];
ry(-2.794549747907291) q[2];
ry(0.06763235470859974) q[3];
cx q[2],q[3];
ry(-3.0627872724838636) q[4];
ry(-0.8536861142160044) q[5];
cx q[4],q[5];
ry(-0.14492248908235386) q[4];
ry(1.1942314398710527) q[5];
cx q[4],q[5];
ry(2.6855716166531254) q[6];
ry(-1.6828859249405737) q[7];
cx q[6],q[7];
ry(-1.9066691738257107) q[6];
ry(2.4438332362243305) q[7];
cx q[6],q[7];
ry(-1.6687132477732158) q[0];
ry(1.0628993343891837) q[2];
cx q[0],q[2];
ry(0.12143289468612685) q[0];
ry(-2.654947936978158) q[2];
cx q[0],q[2];
ry(0.17507608816603387) q[2];
ry(-1.4338379769715033) q[4];
cx q[2],q[4];
ry(0.6086211009461827) q[2];
ry(2.229572561175754) q[4];
cx q[2],q[4];
ry(-2.570388050461486) q[4];
ry(-1.279750289702773) q[6];
cx q[4],q[6];
ry(-0.8036176747884163) q[4];
ry(-1.1471431725637526) q[6];
cx q[4],q[6];
ry(-0.9590502234339645) q[1];
ry(0.7694035059102103) q[3];
cx q[1],q[3];
ry(2.2822142852062135) q[1];
ry(-0.2070626896491845) q[3];
cx q[1],q[3];
ry(0.38579255096301424) q[3];
ry(-2.814131278536708) q[5];
cx q[3],q[5];
ry(-0.884710189912929) q[3];
ry(0.11594122014038177) q[5];
cx q[3],q[5];
ry(-0.14050856345058088) q[5];
ry(2.9453023561403) q[7];
cx q[5],q[7];
ry(0.9169061047519209) q[5];
ry(1.8888141003027041) q[7];
cx q[5],q[7];
ry(1.235914060552492) q[0];
ry(2.7776838404784465) q[3];
cx q[0],q[3];
ry(-2.812182807075349) q[0];
ry(0.990476143522157) q[3];
cx q[0],q[3];
ry(1.124469826243896) q[1];
ry(1.9322214221578176) q[2];
cx q[1],q[2];
ry(1.004725979552017) q[1];
ry(2.7230742579130545) q[2];
cx q[1],q[2];
ry(-1.1462932655519156) q[2];
ry(0.9883917636685364) q[5];
cx q[2],q[5];
ry(0.6118950135602287) q[2];
ry(1.1510283849002156) q[5];
cx q[2],q[5];
ry(2.882054369394872) q[3];
ry(-0.2755917930472102) q[4];
cx q[3],q[4];
ry(-2.3654629055284433) q[3];
ry(-1.7626227069672429) q[4];
cx q[3],q[4];
ry(-2.8381052219311957) q[4];
ry(-0.1292433162797284) q[7];
cx q[4],q[7];
ry(-1.8499022434045342) q[4];
ry(3.090261449192449) q[7];
cx q[4],q[7];
ry(2.014805867256443) q[5];
ry(2.1326346340162567) q[6];
cx q[5],q[6];
ry(-3.1086925553522504) q[5];
ry(1.1923137428732495) q[6];
cx q[5],q[6];
ry(-2.6301216014668864) q[0];
ry(-2.476393527037818) q[1];
cx q[0],q[1];
ry(-0.5139093494434182) q[0];
ry(2.2901724064010005) q[1];
cx q[0],q[1];
ry(1.615291633683186) q[2];
ry(-1.3375223584698317) q[3];
cx q[2],q[3];
ry(0.9425639898421848) q[2];
ry(-3.1108584093071943) q[3];
cx q[2],q[3];
ry(-0.5998820142315653) q[4];
ry(-0.869621244587772) q[5];
cx q[4],q[5];
ry(2.037747561640088) q[4];
ry(-1.0569932104297752) q[5];
cx q[4],q[5];
ry(2.751615437173096) q[6];
ry(2.043955701944091) q[7];
cx q[6],q[7];
ry(-2.8460985080641836) q[6];
ry(2.7036655211165863) q[7];
cx q[6],q[7];
ry(2.6848473904159422) q[0];
ry(-0.4148009897270777) q[2];
cx q[0],q[2];
ry(2.0472287776979523) q[0];
ry(-1.3664162527750991) q[2];
cx q[0],q[2];
ry(-0.04585317774244247) q[2];
ry(1.3649346313433128) q[4];
cx q[2],q[4];
ry(-1.114612179514046) q[2];
ry(2.813966286625844) q[4];
cx q[2],q[4];
ry(1.9143548855529626) q[4];
ry(0.3581679971905176) q[6];
cx q[4],q[6];
ry(1.5290239310181246) q[4];
ry(-0.3987196787057388) q[6];
cx q[4],q[6];
ry(0.29971064985420526) q[1];
ry(2.3899158122256687) q[3];
cx q[1],q[3];
ry(3.02164903690441) q[1];
ry(-2.430062835315396) q[3];
cx q[1],q[3];
ry(2.161096714953706) q[3];
ry(3.0175784016177873) q[5];
cx q[3],q[5];
ry(-1.8208859997317717) q[3];
ry(-0.8650667636729205) q[5];
cx q[3],q[5];
ry(-1.7838713207303876) q[5];
ry(-1.504015594502941) q[7];
cx q[5],q[7];
ry(-1.8095407099904515) q[5];
ry(-0.7461671041420246) q[7];
cx q[5],q[7];
ry(2.7776503508746164) q[0];
ry(-0.6264340807721501) q[3];
cx q[0],q[3];
ry(0.5339736105945976) q[0];
ry(2.7529827922417986) q[3];
cx q[0],q[3];
ry(-0.5548517695083014) q[1];
ry(0.8133951452052904) q[2];
cx q[1],q[2];
ry(0.09123059474509512) q[1];
ry(1.4676282536909628) q[2];
cx q[1],q[2];
ry(-3.0017684587239764) q[2];
ry(-1.5641623786796783) q[5];
cx q[2],q[5];
ry(2.647710116762291) q[2];
ry(-0.7376554244527709) q[5];
cx q[2],q[5];
ry(1.0135458863440299) q[3];
ry(2.5453726691566945) q[4];
cx q[3],q[4];
ry(1.0365001768256594) q[3];
ry(-2.2727603444655315) q[4];
cx q[3],q[4];
ry(1.3857458119424033) q[4];
ry(0.09068354995284597) q[7];
cx q[4],q[7];
ry(1.6965085287645094) q[4];
ry(0.06494923674450255) q[7];
cx q[4],q[7];
ry(-3.0963772342605007) q[5];
ry(2.4843810310776173) q[6];
cx q[5],q[6];
ry(-1.1739810814955414) q[5];
ry(0.5289212978541107) q[6];
cx q[5],q[6];
ry(0.884655678892401) q[0];
ry(1.2049340872443657) q[1];
cx q[0],q[1];
ry(-2.629343015016214) q[0];
ry(1.4336923525606566) q[1];
cx q[0],q[1];
ry(1.495730586728902) q[2];
ry(2.0429169767902273) q[3];
cx q[2],q[3];
ry(0.4309819988699797) q[2];
ry(-2.7929082349812457) q[3];
cx q[2],q[3];
ry(2.6765536685839773) q[4];
ry(0.949125391747818) q[5];
cx q[4],q[5];
ry(2.3323084315043676) q[4];
ry(-0.6653979244956822) q[5];
cx q[4],q[5];
ry(1.7188656898330041) q[6];
ry(-1.3888101053055317) q[7];
cx q[6],q[7];
ry(0.12230814013214435) q[6];
ry(1.2868813926158662) q[7];
cx q[6],q[7];
ry(2.0935370774712077) q[0];
ry(0.638069844017054) q[2];
cx q[0],q[2];
ry(-0.035511097610252576) q[0];
ry(0.953456518370003) q[2];
cx q[0],q[2];
ry(1.6142033151098074) q[2];
ry(-0.9977306594482653) q[4];
cx q[2],q[4];
ry(2.7511709021547337) q[2];
ry(-2.7212995555883994) q[4];
cx q[2],q[4];
ry(1.4735687366667545) q[4];
ry(1.5520914893515059) q[6];
cx q[4],q[6];
ry(1.3061750444463902) q[4];
ry(1.86285877388681) q[6];
cx q[4],q[6];
ry(2.9894987860689444) q[1];
ry(-1.271137377352508) q[3];
cx q[1],q[3];
ry(-0.4127261974869093) q[1];
ry(-1.623007694930572) q[3];
cx q[1],q[3];
ry(-0.4065634349225409) q[3];
ry(-1.9202091899857496) q[5];
cx q[3],q[5];
ry(2.7147630596190573) q[3];
ry(1.9562875217084938) q[5];
cx q[3],q[5];
ry(1.214226289310861) q[5];
ry(2.9517975023022056) q[7];
cx q[5],q[7];
ry(-0.2417362971729817) q[5];
ry(-1.0759377066596292) q[7];
cx q[5],q[7];
ry(2.833419301363118) q[0];
ry(2.7447132208862923) q[3];
cx q[0],q[3];
ry(-1.7354491653640531) q[0];
ry(-0.06225871638412794) q[3];
cx q[0],q[3];
ry(1.7425123204199595) q[1];
ry(-0.5045227723648712) q[2];
cx q[1],q[2];
ry(-1.509698418547672) q[1];
ry(-0.0019004211509755504) q[2];
cx q[1],q[2];
ry(-1.1673788673633787) q[2];
ry(1.0255295420390036) q[5];
cx q[2],q[5];
ry(1.718393826906083) q[2];
ry(1.648592309631361) q[5];
cx q[2],q[5];
ry(-1.640926921660901) q[3];
ry(2.4265579785003832) q[4];
cx q[3],q[4];
ry(2.1573965211150097) q[3];
ry(-0.7155629143119285) q[4];
cx q[3],q[4];
ry(-0.40784969693278317) q[4];
ry(-2.008505222906402) q[7];
cx q[4],q[7];
ry(-3.127432955266718) q[4];
ry(-1.4430822151691496) q[7];
cx q[4],q[7];
ry(-1.1041306113121572) q[5];
ry(1.309009513580049) q[6];
cx q[5],q[6];
ry(2.5782439620732855) q[5];
ry(-0.07704260256779244) q[6];
cx q[5],q[6];
ry(-0.6924071582335621) q[0];
ry(0.7633541206174002) q[1];
cx q[0],q[1];
ry(-2.105263156589984) q[0];
ry(-2.9106043831378066) q[1];
cx q[0],q[1];
ry(3.1353783724898387) q[2];
ry(-2.6552569902904537) q[3];
cx q[2],q[3];
ry(1.379211453154274) q[2];
ry(1.8160527725878313) q[3];
cx q[2],q[3];
ry(0.579651501328101) q[4];
ry(-1.0555364464492776) q[5];
cx q[4],q[5];
ry(0.7620176306086206) q[4];
ry(1.4842903265918244) q[5];
cx q[4],q[5];
ry(0.7924991221595621) q[6];
ry(-2.026399032100262) q[7];
cx q[6],q[7];
ry(-1.624825229509175) q[6];
ry(-1.6658021378873293) q[7];
cx q[6],q[7];
ry(2.1473032370129244) q[0];
ry(1.278230332695371) q[2];
cx q[0],q[2];
ry(-0.5920673771041862) q[0];
ry(-1.168890023949823) q[2];
cx q[0],q[2];
ry(-1.7112702809604505) q[2];
ry(-0.03260410004705783) q[4];
cx q[2],q[4];
ry(-2.7830191952124634) q[2];
ry(-0.5630288625042432) q[4];
cx q[2],q[4];
ry(-1.646163143279196) q[4];
ry(-0.8655806874272978) q[6];
cx q[4],q[6];
ry(-0.44482951850446345) q[4];
ry(2.2034477221751914) q[6];
cx q[4],q[6];
ry(0.0340080188198808) q[1];
ry(-1.2857887790376599) q[3];
cx q[1],q[3];
ry(2.1017469700053306) q[1];
ry(-0.07615110553403692) q[3];
cx q[1],q[3];
ry(0.3865878664880258) q[3];
ry(-0.4753418343962901) q[5];
cx q[3],q[5];
ry(-2.1616037379376367) q[3];
ry(-2.4433980796575074) q[5];
cx q[3],q[5];
ry(1.6228888094806013) q[5];
ry(-3.1084429605097954) q[7];
cx q[5],q[7];
ry(1.7686885153354073) q[5];
ry(-1.6652879833213774) q[7];
cx q[5],q[7];
ry(3.0684020229738462) q[0];
ry(2.8857250894899678) q[3];
cx q[0],q[3];
ry(-0.6672436913212569) q[0];
ry(-1.1755434197398593) q[3];
cx q[0],q[3];
ry(-2.6897205306112726) q[1];
ry(2.0465659438310126) q[2];
cx q[1],q[2];
ry(-0.42774696819301194) q[1];
ry(0.23642895517197882) q[2];
cx q[1],q[2];
ry(1.115411313672645) q[2];
ry(3.1210174461708164) q[5];
cx q[2],q[5];
ry(-2.790245062280825) q[2];
ry(-0.16824819075185893) q[5];
cx q[2],q[5];
ry(-3.129719149073316) q[3];
ry(2.360156455088914) q[4];
cx q[3],q[4];
ry(2.5837767084776977) q[3];
ry(-0.6652177007568413) q[4];
cx q[3],q[4];
ry(0.7765453347516794) q[4];
ry(2.6610296112203544) q[7];
cx q[4],q[7];
ry(-1.1542060909799572) q[4];
ry(1.5789201148835161) q[7];
cx q[4],q[7];
ry(-2.883976288531596) q[5];
ry(2.083929491197829) q[6];
cx q[5],q[6];
ry(0.9644859986483442) q[5];
ry(1.0340805466084353) q[6];
cx q[5],q[6];
ry(2.789878620418595) q[0];
ry(-2.140135567045209) q[1];
cx q[0],q[1];
ry(3.0376390013582735) q[0];
ry(2.6732297316641946) q[1];
cx q[0],q[1];
ry(1.8798617279127738) q[2];
ry(0.5863137902290859) q[3];
cx q[2],q[3];
ry(0.8779334566147794) q[2];
ry(-1.5595971633335601) q[3];
cx q[2],q[3];
ry(2.96072774544379) q[4];
ry(1.2412330264674127) q[5];
cx q[4],q[5];
ry(-2.993720960630088) q[4];
ry(-0.5361520407873852) q[5];
cx q[4],q[5];
ry(2.340127157173656) q[6];
ry(0.3111532435808982) q[7];
cx q[6],q[7];
ry(2.7518789032155957) q[6];
ry(0.3264818860249115) q[7];
cx q[6],q[7];
ry(-2.7925477382168014) q[0];
ry(2.201490120124287) q[2];
cx q[0],q[2];
ry(2.5442595045522576) q[0];
ry(0.5660404882364222) q[2];
cx q[0],q[2];
ry(-1.3217083670048817) q[2];
ry(0.5149484997616912) q[4];
cx q[2],q[4];
ry(0.25522470081516263) q[2];
ry(-2.79494361260819) q[4];
cx q[2],q[4];
ry(0.17399532222307812) q[4];
ry(0.5734909718798811) q[6];
cx q[4],q[6];
ry(0.7046523318945179) q[4];
ry(-1.167903449737036) q[6];
cx q[4],q[6];
ry(-3.0286790507789716) q[1];
ry(-0.9803067399740008) q[3];
cx q[1],q[3];
ry(-0.5284389477128366) q[1];
ry(-0.8605242124860389) q[3];
cx q[1],q[3];
ry(-1.4843155393757872) q[3];
ry(0.9155387254627813) q[5];
cx q[3],q[5];
ry(-1.6088139187301493) q[3];
ry(1.7198551623337437) q[5];
cx q[3],q[5];
ry(3.0760829670238476) q[5];
ry(0.9095895674549199) q[7];
cx q[5],q[7];
ry(1.5133427926943381) q[5];
ry(-1.449522045215665) q[7];
cx q[5],q[7];
ry(1.5356741980731878) q[0];
ry(0.9706912461384531) q[3];
cx q[0],q[3];
ry(-2.9816459830645017) q[0];
ry(2.2598323179629123) q[3];
cx q[0],q[3];
ry(1.2911141951015708) q[1];
ry(-2.4164686512536915) q[2];
cx q[1],q[2];
ry(2.2635662494574134) q[1];
ry(2.7302306069571496) q[2];
cx q[1],q[2];
ry(-1.642803003201705) q[2];
ry(-2.482876574136292) q[5];
cx q[2],q[5];
ry(3.081572903781173) q[2];
ry(0.7347602749853825) q[5];
cx q[2],q[5];
ry(-0.9270921834299655) q[3];
ry(-0.914833492444274) q[4];
cx q[3],q[4];
ry(-2.979364388197325) q[3];
ry(-2.5642801521683514) q[4];
cx q[3],q[4];
ry(0.26540298401606316) q[4];
ry(0.14362644946440245) q[7];
cx q[4],q[7];
ry(0.924569292188234) q[4];
ry(1.7769559947451523) q[7];
cx q[4],q[7];
ry(-1.5658397159392736) q[5];
ry(-2.715566314919032) q[6];
cx q[5],q[6];
ry(-0.8850636119538009) q[5];
ry(0.22387264381996264) q[6];
cx q[5],q[6];
ry(0.33205304047101425) q[0];
ry(-0.43245705965132686) q[1];
cx q[0],q[1];
ry(0.9698466810164525) q[0];
ry(3.060755956887448) q[1];
cx q[0],q[1];
ry(0.5493481744098636) q[2];
ry(-1.5981124284335992) q[3];
cx q[2],q[3];
ry(-2.5533067450579336) q[2];
ry(-2.4869822598006226) q[3];
cx q[2],q[3];
ry(1.9758788488702592) q[4];
ry(0.7226639132573713) q[5];
cx q[4],q[5];
ry(-1.0476080521985545) q[4];
ry(-2.0747440029174076) q[5];
cx q[4],q[5];
ry(2.7033267916338777) q[6];
ry(-2.4485321183199886) q[7];
cx q[6],q[7];
ry(1.0275517019219078) q[6];
ry(0.985120610759901) q[7];
cx q[6],q[7];
ry(-1.0228965362514941) q[0];
ry(-0.8693746484379927) q[2];
cx q[0],q[2];
ry(-0.4979731207976501) q[0];
ry(-1.3800295545271715) q[2];
cx q[0],q[2];
ry(-0.7053661143846401) q[2];
ry(1.3815537830563678) q[4];
cx q[2],q[4];
ry(-2.190719353419655) q[2];
ry(-2.740210736752365) q[4];
cx q[2],q[4];
ry(-2.4868477450225797) q[4];
ry(2.5555647421584866) q[6];
cx q[4],q[6];
ry(1.1296972410411508) q[4];
ry(0.945327144915793) q[6];
cx q[4],q[6];
ry(1.777666177184027) q[1];
ry(2.2737822533042804) q[3];
cx q[1],q[3];
ry(0.3565534173326821) q[1];
ry(2.447706647961779) q[3];
cx q[1],q[3];
ry(-0.7233363887639896) q[3];
ry(-0.5635902967519052) q[5];
cx q[3],q[5];
ry(0.035589427772014376) q[3];
ry(-0.40436644139103617) q[5];
cx q[3],q[5];
ry(3.0373390383174144) q[5];
ry(1.4864057410016116) q[7];
cx q[5],q[7];
ry(-0.3305972775160395) q[5];
ry(-1.662842097487253) q[7];
cx q[5],q[7];
ry(-1.318965110280215) q[0];
ry(-1.51154187716062) q[3];
cx q[0],q[3];
ry(-3.1297776774190025) q[0];
ry(-1.1397085587785547) q[3];
cx q[0],q[3];
ry(-2.256930735876276) q[1];
ry(3.1113568702838053) q[2];
cx q[1],q[2];
ry(2.7451136274305092) q[1];
ry(2.463261980439224) q[2];
cx q[1],q[2];
ry(0.1434816440852087) q[2];
ry(-1.643092794271948) q[5];
cx q[2],q[5];
ry(0.6218052476629315) q[2];
ry(1.7948583177685267) q[5];
cx q[2],q[5];
ry(-2.810858636144138) q[3];
ry(-2.8369336680779136) q[4];
cx q[3],q[4];
ry(2.3214453671252575) q[3];
ry(2.9631003047577784) q[4];
cx q[3],q[4];
ry(1.3487035006101147) q[4];
ry(2.7486569345809637) q[7];
cx q[4],q[7];
ry(1.594936945167162) q[4];
ry(2.117145962686366) q[7];
cx q[4],q[7];
ry(-2.1716097427552326) q[5];
ry(0.19800561637599187) q[6];
cx q[5],q[6];
ry(-0.5527580103921406) q[5];
ry(2.2752782408025842) q[6];
cx q[5],q[6];
ry(-2.674834153040705) q[0];
ry(0.3364873653637419) q[1];
cx q[0],q[1];
ry(0.6845990394366019) q[0];
ry(-2.699764682547938) q[1];
cx q[0],q[1];
ry(-2.8236171211143257) q[2];
ry(-2.8604499361378672) q[3];
cx q[2],q[3];
ry(-0.038759679061256706) q[2];
ry(2.321963084662444) q[3];
cx q[2],q[3];
ry(1.2099298046489295) q[4];
ry(-0.5568652288402118) q[5];
cx q[4],q[5];
ry(1.121341477458298) q[4];
ry(-1.086505733301819) q[5];
cx q[4],q[5];
ry(2.8291716708099792) q[6];
ry(0.418950087151003) q[7];
cx q[6],q[7];
ry(-2.2429726674470007) q[6];
ry(-0.8221000514637645) q[7];
cx q[6],q[7];
ry(-3.0991655054637204) q[0];
ry(2.2707801573534283) q[2];
cx q[0],q[2];
ry(-0.4594630555254332) q[0];
ry(-3.0087602113472114) q[2];
cx q[0],q[2];
ry(1.3847775414979426) q[2];
ry(-2.4901452708840837) q[4];
cx q[2],q[4];
ry(-1.9563529945531062) q[2];
ry(2.317343149841819) q[4];
cx q[2],q[4];
ry(0.4452759984036492) q[4];
ry(-0.822642897643757) q[6];
cx q[4],q[6];
ry(2.766874187765783) q[4];
ry(2.0462674472751363) q[6];
cx q[4],q[6];
ry(-0.8447965008843539) q[1];
ry(2.061351468323825) q[3];
cx q[1],q[3];
ry(-2.3569142627928956) q[1];
ry(-2.629511465327665) q[3];
cx q[1],q[3];
ry(2.914293086981967) q[3];
ry(3.139360330423813) q[5];
cx q[3],q[5];
ry(-1.4674821766748583) q[3];
ry(-0.5843438096663123) q[5];
cx q[3],q[5];
ry(-0.9213958731149923) q[5];
ry(-2.49435282084528) q[7];
cx q[5],q[7];
ry(-1.6777275849472897) q[5];
ry(1.7041138687907618) q[7];
cx q[5],q[7];
ry(2.5655558715421325) q[0];
ry(-0.9522214838451912) q[3];
cx q[0],q[3];
ry(-1.3375741280600542) q[0];
ry(-1.0240572913662438) q[3];
cx q[0],q[3];
ry(-2.7992721234818205) q[1];
ry(-2.8710582335857304) q[2];
cx q[1],q[2];
ry(1.412633516489516) q[1];
ry(2.500783371278121) q[2];
cx q[1],q[2];
ry(1.612034392530846) q[2];
ry(1.819366265609638) q[5];
cx q[2],q[5];
ry(-2.41839297661402) q[2];
ry(-1.8022366961896215) q[5];
cx q[2],q[5];
ry(-1.2811861559294035) q[3];
ry(1.8115817989576497) q[4];
cx q[3],q[4];
ry(2.8854734209464894) q[3];
ry(2.460361893363487) q[4];
cx q[3],q[4];
ry(-1.576322885020257) q[4];
ry(-0.9808686904932494) q[7];
cx q[4],q[7];
ry(-1.2422544723436784) q[4];
ry(1.9418103216293874) q[7];
cx q[4],q[7];
ry(-1.6672538071334564) q[5];
ry(-1.6171567169565728) q[6];
cx q[5],q[6];
ry(2.5983484898742346) q[5];
ry(0.9714572949916648) q[6];
cx q[5],q[6];
ry(-2.0712261450293923) q[0];
ry(2.2312069085918367) q[1];
cx q[0],q[1];
ry(1.9925822528812878) q[0];
ry(-2.5213925559657344) q[1];
cx q[0],q[1];
ry(-2.6214678411716896) q[2];
ry(1.0986785145419669) q[3];
cx q[2],q[3];
ry(2.699122599623541) q[2];
ry(-1.6953590450557596) q[3];
cx q[2],q[3];
ry(-2.2832218994699542) q[4];
ry(-0.9990726134515662) q[5];
cx q[4],q[5];
ry(2.876611761088153) q[4];
ry(-2.8937709498858983) q[5];
cx q[4],q[5];
ry(-3.090911578769575) q[6];
ry(-2.09205048433921) q[7];
cx q[6],q[7];
ry(-0.38216772852086134) q[6];
ry(-1.43293615160509) q[7];
cx q[6],q[7];
ry(-2.949620257397122) q[0];
ry(-1.514795350004034) q[2];
cx q[0],q[2];
ry(-1.8045383886291728) q[0];
ry(2.641006789101353) q[2];
cx q[0],q[2];
ry(-2.3808480438516475) q[2];
ry(-1.855239001586705) q[4];
cx q[2],q[4];
ry(-1.7182078980017703) q[2];
ry(-3.0662062678994064) q[4];
cx q[2],q[4];
ry(2.6619172696098063) q[4];
ry(-1.4498137813554681) q[6];
cx q[4],q[6];
ry(1.5222849233322773) q[4];
ry(-1.5927110965105715) q[6];
cx q[4],q[6];
ry(-1.992438608060097) q[1];
ry(-0.5622090962027935) q[3];
cx q[1],q[3];
ry(1.6041235476056794) q[1];
ry(1.9549547663228894) q[3];
cx q[1],q[3];
ry(-2.723320424993681) q[3];
ry(0.3182098472068148) q[5];
cx q[3],q[5];
ry(-0.44627693611623975) q[3];
ry(-2.590359143606958) q[5];
cx q[3],q[5];
ry(-1.5572047102224849) q[5];
ry(-3.1081614149203993) q[7];
cx q[5],q[7];
ry(1.3457007341741036) q[5];
ry(0.5215233048050356) q[7];
cx q[5],q[7];
ry(-2.296448039320448) q[0];
ry(2.546947433423991) q[3];
cx q[0],q[3];
ry(-1.3134497268183951) q[0];
ry(-1.3414579840944978) q[3];
cx q[0],q[3];
ry(2.1947281556160156) q[1];
ry(-2.7853958057692862) q[2];
cx q[1],q[2];
ry(2.738979007419706) q[1];
ry(0.4531094546998848) q[2];
cx q[1],q[2];
ry(-0.018367315297303108) q[2];
ry(1.314324460168445) q[5];
cx q[2],q[5];
ry(-0.982450749426397) q[2];
ry(-2.0892640111349685) q[5];
cx q[2],q[5];
ry(-0.9554716966054686) q[3];
ry(-1.025770877726928) q[4];
cx q[3],q[4];
ry(0.721268047284605) q[3];
ry(-0.44587185975947907) q[4];
cx q[3],q[4];
ry(0.2419583696993195) q[4];
ry(-2.0736488237596964) q[7];
cx q[4],q[7];
ry(2.759448606193372) q[4];
ry(-0.6493517747007695) q[7];
cx q[4],q[7];
ry(3.0522164640995855) q[5];
ry(-1.0680625474429517) q[6];
cx q[5],q[6];
ry(-2.792964205624553) q[5];
ry(-2.8172763189529118) q[6];
cx q[5],q[6];
ry(0.1881975991255811) q[0];
ry(-2.939058068316588) q[1];
cx q[0],q[1];
ry(0.08377320494155785) q[0];
ry(0.866655750802783) q[1];
cx q[0],q[1];
ry(2.643207259587087) q[2];
ry(2.889631326749761) q[3];
cx q[2],q[3];
ry(1.1261595583046473) q[2];
ry(-0.7426951995653619) q[3];
cx q[2],q[3];
ry(-1.8295372662451772) q[4];
ry(-2.658532945410743) q[5];
cx q[4],q[5];
ry(-0.5291671986668083) q[4];
ry(0.489726466856419) q[5];
cx q[4],q[5];
ry(-1.835084627474592) q[6];
ry(1.9422127312735222) q[7];
cx q[6],q[7];
ry(-2.3654768829039403) q[6];
ry(-2.3176594406385655) q[7];
cx q[6],q[7];
ry(-2.0200860984176177) q[0];
ry(0.8670954779625432) q[2];
cx q[0],q[2];
ry(-1.8578330823744844) q[0];
ry(1.6374018918034112) q[2];
cx q[0],q[2];
ry(3.108575988585208) q[2];
ry(0.8334302308780336) q[4];
cx q[2],q[4];
ry(-0.8022031211500451) q[2];
ry(-0.7200904402168036) q[4];
cx q[2],q[4];
ry(0.48991866534328715) q[4];
ry(2.1563950900373055) q[6];
cx q[4],q[6];
ry(-2.3029738444530063) q[4];
ry(0.6840299580507292) q[6];
cx q[4],q[6];
ry(-2.1533419503251983) q[1];
ry(-1.7622133112914673) q[3];
cx q[1],q[3];
ry(-3.094733260160699) q[1];
ry(-0.403076480849491) q[3];
cx q[1],q[3];
ry(-0.5008453241228503) q[3];
ry(2.3196709326583345) q[5];
cx q[3],q[5];
ry(-2.8393621377174867) q[3];
ry(-0.7743891629267053) q[5];
cx q[3],q[5];
ry(1.5073903096006656) q[5];
ry(-2.6738261628903537) q[7];
cx q[5],q[7];
ry(-2.0115649933923168) q[5];
ry(3.085232809625062) q[7];
cx q[5],q[7];
ry(-0.05652498163234938) q[0];
ry(-1.080988315250389) q[3];
cx q[0],q[3];
ry(0.24126652515308766) q[0];
ry(0.6538284137976825) q[3];
cx q[0],q[3];
ry(-2.3531985000576734) q[1];
ry(3.123162403571968) q[2];
cx q[1],q[2];
ry(2.048527249589317) q[1];
ry(2.191415108846801) q[2];
cx q[1],q[2];
ry(2.8414138041701062) q[2];
ry(2.6940886178768775) q[5];
cx q[2],q[5];
ry(-1.0280732737926712) q[2];
ry(1.241820422117332) q[5];
cx q[2],q[5];
ry(3.0073353131970335) q[3];
ry(2.9132140006483187) q[4];
cx q[3],q[4];
ry(-2.1870228741299096) q[3];
ry(1.9544819200282868) q[4];
cx q[3],q[4];
ry(0.9911618265285992) q[4];
ry(2.116145691986873) q[7];
cx q[4],q[7];
ry(-2.479353730862456) q[4];
ry(-1.354768560026435) q[7];
cx q[4],q[7];
ry(-2.5018529346746083) q[5];
ry(-2.646015084811469) q[6];
cx q[5],q[6];
ry(2.273371521014122) q[5];
ry(1.2603211281628308) q[6];
cx q[5],q[6];
ry(1.3114477567544667) q[0];
ry(-0.13314774080438965) q[1];
cx q[0],q[1];
ry(-3.042970773268582) q[0];
ry(1.902131761715264) q[1];
cx q[0],q[1];
ry(0.2338793986256288) q[2];
ry(-1.6961625344074884) q[3];
cx q[2],q[3];
ry(1.563718376331575) q[2];
ry(3.020665935257459) q[3];
cx q[2],q[3];
ry(0.7392013774973166) q[4];
ry(-1.0942751944535556) q[5];
cx q[4],q[5];
ry(2.973765058856041) q[4];
ry(0.9258464233231587) q[5];
cx q[4],q[5];
ry(1.663733076786819) q[6];
ry(0.6339152748556298) q[7];
cx q[6],q[7];
ry(-0.957425818616044) q[6];
ry(-1.7959472393334623) q[7];
cx q[6],q[7];
ry(-1.6752042208340816) q[0];
ry(2.1491268697213863) q[2];
cx q[0],q[2];
ry(2.0624641953889116) q[0];
ry(1.6377934023594447) q[2];
cx q[0],q[2];
ry(0.22651938371293767) q[2];
ry(-3.1058979218445986) q[4];
cx q[2],q[4];
ry(-2.669223949972649) q[2];
ry(-1.422969289632388) q[4];
cx q[2],q[4];
ry(-0.8595290432787892) q[4];
ry(2.876067372701182) q[6];
cx q[4],q[6];
ry(2.6139260391635837) q[4];
ry(1.6035096408061456) q[6];
cx q[4],q[6];
ry(0.27594001920555705) q[1];
ry(1.9590890860992023) q[3];
cx q[1],q[3];
ry(3.0722304991728393) q[1];
ry(-0.288179282613231) q[3];
cx q[1],q[3];
ry(-2.449845527358563) q[3];
ry(2.723621237719318) q[5];
cx q[3],q[5];
ry(-2.077981376472838) q[3];
ry(1.0658607215332314) q[5];
cx q[3],q[5];
ry(2.6975111054548186) q[5];
ry(0.5317760951588952) q[7];
cx q[5],q[7];
ry(0.46109330394435144) q[5];
ry(-2.980196089474413) q[7];
cx q[5],q[7];
ry(1.8933912766496168) q[0];
ry(2.335337992877435) q[3];
cx q[0],q[3];
ry(-0.808389528436007) q[0];
ry(-2.9306771938863085) q[3];
cx q[0],q[3];
ry(2.1924545178113446) q[1];
ry(0.3627945225163896) q[2];
cx q[1],q[2];
ry(-1.6370221610484763) q[1];
ry(2.563021169565937) q[2];
cx q[1],q[2];
ry(-1.4107541479077854) q[2];
ry(-2.1169119271041383) q[5];
cx q[2],q[5];
ry(-0.9893121612825357) q[2];
ry(1.3357361623280664) q[5];
cx q[2],q[5];
ry(0.29431663587557827) q[3];
ry(0.8394795082632033) q[4];
cx q[3],q[4];
ry(-1.8149509466839973) q[3];
ry(2.1099026722149405) q[4];
cx q[3],q[4];
ry(-1.7115350557501248) q[4];
ry(-1.0409800290643725) q[7];
cx q[4],q[7];
ry(2.0492554846562996) q[4];
ry(-3.0966690465603786) q[7];
cx q[4],q[7];
ry(0.8659917839464129) q[5];
ry(0.03858565993047325) q[6];
cx q[5],q[6];
ry(0.6930520179409002) q[5];
ry(2.893870625854378) q[6];
cx q[5],q[6];
ry(1.715205512443938) q[0];
ry(-0.9310700707022912) q[1];
cx q[0],q[1];
ry(-1.2533473206919032) q[0];
ry(-2.581553314567889) q[1];
cx q[0],q[1];
ry(0.22153563557160275) q[2];
ry(1.9806999662055762) q[3];
cx q[2],q[3];
ry(-2.8513314895900983) q[2];
ry(0.9577023538905544) q[3];
cx q[2],q[3];
ry(-2.422207779486857) q[4];
ry(1.8640686527055994) q[5];
cx q[4],q[5];
ry(2.1979549377604775) q[4];
ry(1.1580203272220908) q[5];
cx q[4],q[5];
ry(0.5946769838447947) q[6];
ry(1.0814463222345685) q[7];
cx q[6],q[7];
ry(0.3037008709282647) q[6];
ry(-2.2833596330816244) q[7];
cx q[6],q[7];
ry(1.7604168298916596) q[0];
ry(-0.6335578142956735) q[2];
cx q[0],q[2];
ry(-1.5657124914561067) q[0];
ry(1.2376881973720595) q[2];
cx q[0],q[2];
ry(-2.1790203372033607) q[2];
ry(-2.518220473083194) q[4];
cx q[2],q[4];
ry(1.7367577584991585) q[2];
ry(2.791322759106885) q[4];
cx q[2],q[4];
ry(1.1461595549258041) q[4];
ry(0.9071262996108614) q[6];
cx q[4],q[6];
ry(-2.5909709659281344) q[4];
ry(2.411452923505926) q[6];
cx q[4],q[6];
ry(-0.06725048001602271) q[1];
ry(-0.9212423123636664) q[3];
cx q[1],q[3];
ry(0.3185551828584394) q[1];
ry(2.7478328956353555) q[3];
cx q[1],q[3];
ry(0.28377923886367196) q[3];
ry(2.2184148370223364) q[5];
cx q[3],q[5];
ry(0.9818336518422188) q[3];
ry(0.4573041672919143) q[5];
cx q[3],q[5];
ry(-2.66283724232185) q[5];
ry(-0.34774563778589407) q[7];
cx q[5],q[7];
ry(0.7380711387431287) q[5];
ry(-1.2568044620111298) q[7];
cx q[5],q[7];
ry(2.973046536460502) q[0];
ry(0.7382906416182768) q[3];
cx q[0],q[3];
ry(2.558499973466895) q[0];
ry(2.5089314950577473) q[3];
cx q[0],q[3];
ry(0.3415359426162357) q[1];
ry(0.41032773478614154) q[2];
cx q[1],q[2];
ry(-0.590942160756077) q[1];
ry(2.62656672392545) q[2];
cx q[1],q[2];
ry(0.021051971189318334) q[2];
ry(-1.5036990860595303) q[5];
cx q[2],q[5];
ry(0.6801509406214832) q[2];
ry(2.1219911839600742) q[5];
cx q[2],q[5];
ry(1.3763049213186995) q[3];
ry(-1.37973758934009) q[4];
cx q[3],q[4];
ry(-1.3852501771213914) q[3];
ry(-3.077536061978293) q[4];
cx q[3],q[4];
ry(-0.969581018651743) q[4];
ry(-2.137208988047583) q[7];
cx q[4],q[7];
ry(-1.3993000319141098) q[4];
ry(-1.8461492470099738) q[7];
cx q[4],q[7];
ry(0.15448926206546928) q[5];
ry(1.3845880349616442) q[6];
cx q[5],q[6];
ry(0.8067743010626467) q[5];
ry(-1.9369479736145099) q[6];
cx q[5],q[6];
ry(-2.586853670418512) q[0];
ry(2.782629561212064) q[1];
cx q[0],q[1];
ry(1.7070133703100352) q[0];
ry(2.1923067759287855) q[1];
cx q[0],q[1];
ry(-2.631723119418508) q[2];
ry(1.6503633187441487) q[3];
cx q[2],q[3];
ry(-0.9478072757121367) q[2];
ry(-0.43974854793614276) q[3];
cx q[2],q[3];
ry(2.4400069001626905) q[4];
ry(1.0184307214944281) q[5];
cx q[4],q[5];
ry(2.3282360105020734) q[4];
ry(0.7328886109741413) q[5];
cx q[4],q[5];
ry(0.9692390507455738) q[6];
ry(-0.5481467284450201) q[7];
cx q[6],q[7];
ry(1.2248048737934338) q[6];
ry(-2.2556975055043447) q[7];
cx q[6],q[7];
ry(0.5667841175374627) q[0];
ry(1.4923535255139926) q[2];
cx q[0],q[2];
ry(1.9122913724727837) q[0];
ry(-0.10393218633307644) q[2];
cx q[0],q[2];
ry(1.4411063298455273) q[2];
ry(2.8289889355705284) q[4];
cx q[2],q[4];
ry(-0.9473556788454579) q[2];
ry(1.9261395363814682) q[4];
cx q[2],q[4];
ry(0.7277321712682422) q[4];
ry(-0.1883869215096055) q[6];
cx q[4],q[6];
ry(0.9585120028817473) q[4];
ry(0.9233310301264533) q[6];
cx q[4],q[6];
ry(0.6328501938264417) q[1];
ry(-1.4011635810105707) q[3];
cx q[1],q[3];
ry(-0.4814773214961932) q[1];
ry(2.6211092192704455) q[3];
cx q[1],q[3];
ry(-2.023146206874352) q[3];
ry(-1.490607099024878) q[5];
cx q[3],q[5];
ry(0.8084874128291012) q[3];
ry(2.450745247488959) q[5];
cx q[3],q[5];
ry(1.6184806209338678) q[5];
ry(-1.7694825894634567) q[7];
cx q[5],q[7];
ry(-1.5337875862506714) q[5];
ry(-1.6571590250429313) q[7];
cx q[5],q[7];
ry(-0.8883512249419905) q[0];
ry(0.015605007362243609) q[3];
cx q[0],q[3];
ry(2.653609008620081) q[0];
ry(-2.167363352671709) q[3];
cx q[0],q[3];
ry(-2.1825431624671086) q[1];
ry(2.8566865391200107) q[2];
cx q[1],q[2];
ry(3.048917974135207) q[1];
ry(-2.2161247862280566) q[2];
cx q[1],q[2];
ry(-2.9471565498198395) q[2];
ry(-1.5999987891142993) q[5];
cx q[2],q[5];
ry(0.8742574645063997) q[2];
ry(0.5256556648703956) q[5];
cx q[2],q[5];
ry(-1.5119933517513093) q[3];
ry(2.2648552804366195) q[4];
cx q[3],q[4];
ry(1.9345932596393745) q[3];
ry(1.800349026436265) q[4];
cx q[3],q[4];
ry(-2.2393151580477495) q[4];
ry(2.519726122595476) q[7];
cx q[4],q[7];
ry(1.3486121042000334) q[4];
ry(1.7872429145636872) q[7];
cx q[4],q[7];
ry(1.7902315199031058) q[5];
ry(-1.6971019592020997) q[6];
cx q[5],q[6];
ry(2.3863440940919305) q[5];
ry(0.8942859605963768) q[6];
cx q[5],q[6];
ry(2.565637305990957) q[0];
ry(3.0902827030490307) q[1];
ry(1.9848074367144724) q[2];
ry(-1.4578008860917944) q[3];
ry(-2.357046613413464) q[4];
ry(0.1382301376296037) q[5];
ry(-2.534356619498577) q[6];
ry(2.944583126020276) q[7];