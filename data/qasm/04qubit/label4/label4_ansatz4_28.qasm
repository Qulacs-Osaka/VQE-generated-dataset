OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.8346050671361238) q[0];
rz(-0.003079145870630739) q[0];
ry(0.5947312178415645) q[1];
rz(0.10399178784903995) q[1];
ry(-0.9082144954831586) q[2];
rz(1.5042011383502227) q[2];
ry(-1.6645549149670877) q[3];
rz(-2.2217978123929876) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.8953265593028094) q[0];
rz(-1.133517243939922) q[0];
ry(-0.6734335544788657) q[1];
rz(-2.461072949546944) q[1];
ry(0.37108316161001276) q[2];
rz(2.448886792816468) q[2];
ry(0.5380164455493144) q[3];
rz(3.001043827229892) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7542718759966842) q[0];
rz(0.02634196887743066) q[0];
ry(-2.54221756363679) q[1];
rz(-2.1395413680807103) q[1];
ry(0.2027899025435502) q[2];
rz(2.18727677121227) q[2];
ry(-1.496532567315705) q[3];
rz(-2.8792625053754928) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.3757199181913782) q[0];
rz(1.3436417301600807) q[0];
ry(2.186241405139723) q[1];
rz(-2.6681885124652043) q[1];
ry(2.585040944106695) q[2];
rz(-0.4491060967334606) q[2];
ry(0.07294622252661474) q[3];
rz(2.2806365848143986) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.853636719530376) q[0];
rz(-0.20777485599600976) q[0];
ry(-0.3745858883511452) q[1];
rz(-0.22498151136592168) q[1];
ry(0.014333643251899453) q[2];
rz(1.6328705552916807) q[2];
ry(-0.3168806985010626) q[3];
rz(-2.7452864922559885) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.157106974899662) q[0];
rz(-1.5195718696005693) q[0];
ry(2.072467780477864) q[1];
rz(2.5273741796513502) q[1];
ry(-1.574702854688336) q[2];
rz(0.20575782248545177) q[2];
ry(0.8841272859483791) q[3];
rz(-1.2157949989918413) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.259054228858917) q[0];
rz(2.1203930705968213) q[0];
ry(-3.125806216360191) q[1];
rz(-1.0758384707437119) q[1];
ry(1.5596056838001058) q[2];
rz(-2.941448622070863) q[2];
ry(-2.4428915750239844) q[3];
rz(-0.9621454126780941) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.1287011234038067) q[0];
rz(-1.9151552064698771) q[0];
ry(-2.5197902500093132) q[1];
rz(1.6655505243584023) q[1];
ry(-2.6948165376680073) q[2];
rz(-2.218173607855542) q[2];
ry(0.2283632849814006) q[3];
rz(-0.7093507219619761) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.872330428265441) q[0];
rz(-2.1395743233577313) q[0];
ry(-2.825250308445584) q[1];
rz(0.3309470398280302) q[1];
ry(2.141289694729063) q[2];
rz(-1.565001814836536) q[2];
ry(-1.9378344442630633) q[3];
rz(-0.8662413227668464) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.9594808436617943) q[0];
rz(-2.7112715147243445) q[0];
ry(-0.505195960288626) q[1];
rz(0.9374215229184149) q[1];
ry(-2.8857737969495783) q[2];
rz(-0.5089195094287229) q[2];
ry(2.4432156767643276) q[3];
rz(2.9559998154814426) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.062861486042829) q[0];
rz(-0.2854089045395284) q[0];
ry(-0.35461707106762574) q[1];
rz(1.7636842528580905) q[1];
ry(-2.8049705422841464) q[2];
rz(-2.991279986489225) q[2];
ry(-0.02848908084559554) q[3];
rz(2.034411050317005) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.853619833153675) q[0];
rz(0.8177222966231286) q[0];
ry(-0.7420292413672431) q[1];
rz(2.751886451013015) q[1];
ry(-2.7762814521235692) q[2];
rz(2.9750771599597985) q[2];
ry(1.4527120732567353) q[3];
rz(-0.7338100167692405) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.87385831876214) q[0];
rz(2.983183849910732) q[0];
ry(0.9102551152676231) q[1];
rz(1.6558493916381096) q[1];
ry(-0.28942220736621826) q[2];
rz(-2.2935492474891714) q[2];
ry(2.5621185960518673) q[3];
rz(2.802552222504956) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.4803420493832419) q[0];
rz(2.8669582402860914) q[0];
ry(0.030964546054173696) q[1];
rz(2.329989600559325) q[1];
ry(0.7049706872509365) q[2];
rz(2.354862049679922) q[2];
ry(-2.1015872935980133) q[3];
rz(-1.5075978856679324) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.725877496632185) q[0];
rz(-2.1304414412874597) q[0];
ry(-1.5908215334873619) q[1];
rz(-1.1261451567723286) q[1];
ry(1.0087707973778084) q[2];
rz(0.0021728980738611133) q[2];
ry(-0.7265556289633633) q[3];
rz(-0.2695876461441867) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.781598004529015) q[0];
rz(-0.6328128570909799) q[0];
ry(1.4069961206994324) q[1];
rz(-2.8343178502631323) q[1];
ry(3.0199760151430874) q[2];
rz(-1.149090593483976) q[2];
ry(-2.8728179957901703) q[3];
rz(0.6889185584783907) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.0797573564212115) q[0];
rz(-2.8908032137251904) q[0];
ry(2.1036796335941528) q[1];
rz(-2.047865251015243) q[1];
ry(0.7458656063723081) q[2];
rz(-0.23091566649071626) q[2];
ry(3.0204179907392503) q[3];
rz(-0.30227872450198756) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.3374856666121172) q[0];
rz(-2.462907545256812) q[0];
ry(2.62275916684239) q[1];
rz(-2.342653921795604) q[1];
ry(2.903490569844743) q[2];
rz(0.9878040226278154) q[2];
ry(-0.7596510867242642) q[3];
rz(-2.743655564757305) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.1435966144427074) q[0];
rz(1.7440277708631253) q[0];
ry(-0.9097573037483929) q[1];
rz(3.0336195438268674) q[1];
ry(2.1435854325834063) q[2];
rz(1.8003930291591703) q[2];
ry(0.8554329578873795) q[3];
rz(2.6785436020420392) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.6748508767044612) q[0];
rz(-2.5728412716980897) q[0];
ry(1.0091341957817779) q[1];
rz(0.49279627947533966) q[1];
ry(-2.7598312940278578) q[2];
rz(-0.21588851464210101) q[2];
ry(-2.9422596354997226) q[3];
rz(2.8043750487374046) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7116820568280362) q[0];
rz(-1.0972488000416183) q[0];
ry(2.3633847626239732) q[1];
rz(-0.18915330756381812) q[1];
ry(-2.1245637370859605) q[2];
rz(1.54557850275474) q[2];
ry(-0.43061559501852487) q[3];
rz(-1.6977662454579985) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.756078549567566) q[0];
rz(-0.9423611741789482) q[0];
ry(-2.1923619939217334) q[1];
rz(-0.037425077764032366) q[1];
ry(-0.5850837450769677) q[2];
rz(-1.6739797215432883) q[2];
ry(0.19955334352474097) q[3];
rz(-1.285157717001988) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.31587652970204816) q[0];
rz(-1.8583084732913293) q[0];
ry(-1.9664450050768574) q[1];
rz(-0.27810839476177124) q[1];
ry(1.285931561228786) q[2];
rz(-0.23753522722627185) q[2];
ry(2.0691165552724775) q[3];
rz(2.040352412448798) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.1384920704436903) q[0];
rz(-1.9839178533332396) q[0];
ry(0.48707818131989455) q[1];
rz(-2.302948676159256) q[1];
ry(-1.6173730882979873) q[2];
rz(-1.352671130671836) q[2];
ry(-0.08218057361398481) q[3];
rz(-0.49923826761126994) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7503951865178635) q[0];
rz(-2.7897476929299234) q[0];
ry(-0.4250541677659459) q[1];
rz(-0.8208178890699109) q[1];
ry(-1.4019542863118195) q[2];
rz(0.8263576486366819) q[2];
ry(-1.862986572232404) q[3];
rz(-2.6408118819671715) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.038965773711449) q[0];
rz(-3.1071081057769128) q[0];
ry(-1.6089999380125075) q[1];
rz(2.6845638193121784) q[1];
ry(-0.6949030143911877) q[2];
rz(-1.2927279044749582) q[2];
ry(-2.1165524371910545) q[3];
rz(1.7015798098300374) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.815698975192706) q[0];
rz(-1.0401633428247896) q[0];
ry(0.3761846031150613) q[1];
rz(-0.8645861130612326) q[1];
ry(-0.22039609736696253) q[2];
rz(-2.678715816462576) q[2];
ry(-2.2608208621909354) q[3];
rz(-0.09910423050829922) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.428652865713141) q[0];
rz(-1.4707948666529855) q[0];
ry(-1.8976308797575863) q[1];
rz(-1.176613948818641) q[1];
ry(0.7758458093800514) q[2];
rz(-1.1766286313255916) q[2];
ry(-1.9683164483833178) q[3];
rz(-1.5585572986057308) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.3024810548623935) q[0];
rz(2.118078103623807) q[0];
ry(2.9817942464957903) q[1];
rz(-1.9734577828398379) q[1];
ry(2.0355076106229393) q[2];
rz(1.41547888847178) q[2];
ry(-2.9752469794922116) q[3];
rz(0.41199374242784864) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.715705038528278) q[0];
rz(-0.03265546408839389) q[0];
ry(0.7235425507197633) q[1];
rz(0.8031551840134996) q[1];
ry(2.246242710035682) q[2];
rz(1.1051874472416396) q[2];
ry(-2.128089658226198) q[3];
rz(2.6773453890601795) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.2073481920243685) q[0];
rz(0.2560329534478232) q[0];
ry(0.4568809170099907) q[1];
rz(3.0358059020926396) q[1];
ry(0.7977594009007304) q[2];
rz(0.34806627562374715) q[2];
ry(2.8295825586819627) q[3];
rz(-0.05453617726903791) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.475254972266838) q[0];
rz(-1.9572575219498531) q[0];
ry(1.885655479789997) q[1];
rz(1.1546065496402915) q[1];
ry(-2.8104792824025804) q[2];
rz(-1.4301895000182885) q[2];
ry(-1.0380566337529702) q[3];
rz(-0.8728275420643486) q[3];