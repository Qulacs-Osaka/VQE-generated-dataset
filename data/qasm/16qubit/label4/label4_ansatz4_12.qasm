OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-3.1107327924066626) q[0];
rz(-2.511692882251318) q[0];
ry(3.061687070328537) q[1];
rz(0.6216041270570039) q[1];
ry(2.025531084631951) q[2];
rz(0.7341521283744248) q[2];
ry(1.54084309573167) q[3];
rz(-0.5299454513065021) q[3];
ry(1.5137863670628682) q[4];
rz(-0.02078212104576327) q[4];
ry(3.1181931336764843) q[5];
rz(1.2142601560311277) q[5];
ry(1.5712412638438134) q[6];
rz(0.33085320389700185) q[6];
ry(-1.5707819428651957) q[7];
rz(-1.166285230440595) q[7];
ry(-0.5111490464382347) q[8];
rz(1.580022458273863) q[8];
ry(7.632195712177622e-05) q[9];
rz(2.1591045312516544) q[9];
ry(3.141535140220993) q[10];
rz(0.6789602092492111) q[10];
ry(-1.5705469648957804) q[11];
rz(-2.7180030354117504) q[11];
ry(1.3911177806239687) q[12];
rz(-1.5729761416164827) q[12];
ry(-1.5801476610530347) q[13];
rz(-2.9749826794186673) q[13];
ry(3.073404172294955) q[14];
rz(2.9418527959496) q[14];
ry(0.0020618324422674306) q[15];
rz(-1.746950471286154) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5160100912180647) q[0];
rz(-2.940196395776852) q[0];
ry(1.087383140221655) q[1];
rz(-2.3617643188304163) q[1];
ry(0.00023349607569613835) q[2];
rz(-1.8407542824881507) q[2];
ry(3.1408445164078573) q[3];
rz(1.036745783685215) q[3];
ry(3.1008475622498306) q[4];
rz(3.1266942256592603) q[4];
ry(0.0022705265934499508) q[5];
rz(1.8187963197655321) q[5];
ry(-0.044619745301876054) q[6];
rz(0.7321759186783137) q[6];
ry(-0.0005613227208344197) q[7];
rz(-0.40562110544721536) q[7];
ry(-1.5710182583310632) q[8];
rz(-1.4205909806738237) q[8];
ry(-3.1413789387033964) q[9];
rz(1.4147709738323098) q[9];
ry(-3.1414175974360883) q[10];
rz(0.18702204911415699) q[10];
ry(-3.1413997075572904) q[11];
rz(3.086565744754357) q[11];
ry(0.7210408936160606) q[12];
rz(-1.336827075455635) q[12];
ry(0.01134979016570181) q[13];
rz(1.7571158275699246) q[13];
ry(-0.0033123451909276014) q[14];
rz(-2.75431753893079) q[14];
ry(-0.001638923780498648) q[15];
rz(-0.9111652142614949) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5920137257364901) q[0];
rz(0.24145173289314523) q[0];
ry(1.5941756288505684) q[1];
rz(0.045600976817200066) q[1];
ry(3.140226338272148) q[2];
rz(-1.7181969890856035) q[2];
ry(1.471964785741531) q[3];
rz(-1.1731221879940872) q[3];
ry(-1.2296249153440744) q[4];
rz(1.5520336549703897) q[4];
ry(1.546296432467619) q[5];
rz(-3.0851415154482162) q[5];
ry(-1.070134578844442) q[6];
rz(2.2576680939135647) q[6];
ry(-2.129253654253155) q[7];
rz(0.7602209096037194) q[7];
ry(0.00011543179771862515) q[8];
rz(2.9864164604377677) q[8];
ry(3.141285669640664) q[9];
rz(-0.8382267027964705) q[9];
ry(-1.5711426702872284) q[10];
rz(2.0526911560216945) q[10];
ry(0.0004604896946664994) q[11];
rz(0.47034002662016544) q[11];
ry(2.874125277330899) q[12];
rz(1.7721115246448402) q[12];
ry(-3.104966299297807) q[13];
rz(1.9229106951318435) q[13];
ry(-0.04428522228105423) q[14];
rz(2.869992507777837) q[14];
ry(3.1364593882869016) q[15];
rz(0.9078508656858839) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5015864618282162) q[0];
rz(1.7762757100486162) q[0];
ry(2.0292357908406133) q[1];
rz(0.3816577614064461) q[1];
ry(3.139754263234602) q[2];
rz(-2.173974764965101) q[2];
ry(0.0010945882279536079) q[3];
rz(1.1678319186684645) q[3];
ry(1.5758475477243739) q[4];
rz(-1.7679603392358265) q[4];
ry(-1.5483637872398788) q[5];
rz(-1.5764351290685452) q[5];
ry(-3.1404349020150453) q[6];
rz(-2.137517581916076) q[6];
ry(-3.1395058846375017) q[7];
rz(0.8201666913916298) q[7];
ry(-1.57839908233808) q[8];
rz(-1.570831259386527) q[8];
ry(-1.570976265129001) q[9];
rz(-0.001231090806009269) q[9];
ry(-3.0796418722335337e-05) q[10];
rz(2.525480900959982) q[10];
ry(0.023782664312216466) q[11];
rz(-0.007028815265658047) q[11];
ry(1.5915392732952869) q[12];
rz(-2.3654297701577773) q[12];
ry(-0.9451581019632872) q[13];
rz(-3.1407821476730446) q[13];
ry(3.138213063383451) q[14];
rz(-1.4371496541308293) q[14];
ry(3.14100497415956) q[15];
rz(-1.721167334476367) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.2975719580412255) q[0];
rz(1.9188070325441844) q[0];
ry(-2.0505907818523497) q[1];
rz(-1.9963095460598324) q[1];
ry(-1.5704107303498167) q[2];
rz(3.1299921567466145) q[2];
ry(-1.5726684600510858) q[3];
rz(0.043639916572856007) q[3];
ry(-3.108626653537274) q[4];
rz(2.9378656881005156) q[4];
ry(-2.8586908590721696) q[5];
rz(-0.005165595884023681) q[5];
ry(-3.1415275536702656) q[6];
rz(-2.551817528860203) q[6];
ry(-3.1412450924135094) q[7];
rz(-3.0814489193209584) q[7];
ry(-1.5708704974648664) q[8];
rz(1.5789791447564236) q[8];
ry(1.5706718913371112) q[9];
rz(0.8040871324194858) q[9];
ry(0.0007259237978898271) q[10];
rz(-3.070061305810985) q[10];
ry(3.141468366907031) q[11];
rz(0.06936135033643968) q[11];
ry(-1.5854559536048645) q[12];
rz(-1.992722136529311) q[12];
ry(1.5709275550922346) q[13];
rz(-1.568685000710632) q[13];
ry(-1.8277679993608187) q[14];
rz(1.2486836400120382) q[14];
ry(0.8931518579705344) q[15];
rz(-3.114073927449449) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(3.138282511447638) q[0];
rz(-2.7881206744515503) q[0];
ry(0.007539801797446488) q[1];
rz(0.6205191427335178) q[1];
ry(-0.02420360709203544) q[2];
rz(0.013106661423017309) q[2];
ry(3.1410265875071994) q[3];
rz(-3.0979708677804054) q[3];
ry(-1.5525373528704396) q[4];
rz(-1.5690281755972144) q[4];
ry(-1.6160679091994101) q[5];
rz(1.9152122169565164) q[5];
ry(-1.575919374946074) q[6];
rz(-1.5688929002021155) q[6];
ry(-1.5708492067620794) q[7];
rz(1.3270458602104176) q[7];
ry(0.15725936359448817) q[8];
rz(-1.4491983821528303) q[8];
ry(3.1360704185027224) q[9];
rz(-2.5168807906294632) q[9];
ry(-1.6007267888117598) q[10];
rz(-3.135985045275189) q[10];
ry(1.60454566886606) q[11];
rz(2.042453506832366) q[11];
ry(3.141401248273203) q[12];
rz(2.4677936192159597) q[12];
ry(-0.03588203922493082) q[13];
rz(1.2161499124644772) q[13];
ry(1.428380224400335) q[14];
rz(0.08627257270321499) q[14];
ry(-1.6010715489987035) q[15];
rz(-0.04956433098067423) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5056333592800517) q[0];
rz(2.5492916477538636) q[0];
ry(1.3956666469929386) q[1];
rz(1.1041786452303337) q[1];
ry(-1.61432265496583) q[2];
rz(0.025247134824902595) q[2];
ry(1.5325422522512504) q[3];
rz(-0.0016999358211886376) q[3];
ry(1.5708760197637943) q[4];
rz(1.5708684405271873) q[4];
ry(1.5703060026325635) q[5];
rz(-1.8617204298064698) q[5];
ry(0.08014708581430696) q[6];
rz(-1.1734764186195665) q[6];
ry(3.14130354876811) q[7];
rz(1.331645471436672) q[7];
ry(0.0003500682821906409) q[8];
rz(1.4411884365147056) q[8];
ry(-3.1415517535943134) q[9];
rz(-0.5513936444121438) q[9];
ry(-3.118605884902067) q[10];
rz(1.1186872213733579) q[10];
ry(3.1414646730181763) q[11];
rz(0.46805730939819057) q[11];
ry(-1.2277597121190807e-05) q[12];
rz(1.04626764526354) q[12];
ry(-0.00011117270800313702) q[13];
rz(-1.5319975354629678) q[13];
ry(-1.536314805195224) q[14];
rz(-0.12751457807131958) q[14];
ry(1.3454063905392548) q[15];
rz(1.1968360252356787) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.8346846307460454) q[0];
rz(2.0912756896507525) q[0];
ry(1.7326484028227283) q[1];
rz(-1.320101576540278) q[1];
ry(-1.5695408649529425) q[2];
rz(1.685186139839657) q[2];
ry(-1.5677890481213215) q[3];
rz(0.0626039674011123) q[3];
ry(-1.575047315528118) q[4];
rz(0.16175619067171887) q[4];
ry(3.1271822110114367) q[5];
rz(2.84899907775438) q[5];
ry(1.5704276817203735) q[6];
rz(1.5718548729140212) q[6];
ry(-1.5707677782138767) q[7];
rz(3.141245109193167) q[7];
ry(-1.5754136363806117) q[8];
rz(1.6352872209564482) q[8];
ry(-3.1374558992567336) q[9];
rz(-1.4183926467519914) q[9];
ry(3.0740484882911354) q[10];
rz(-2.0290482753574484) q[10];
ry(1.4827223562882115) q[11];
rz(1.6044524417683643) q[11];
ry(0.0009119765456979292) q[12];
rz(2.3881819671284688) q[12];
ry(1.606565342744339) q[13];
rz(-1.9345842914275257) q[13];
ry(-0.8779711911928109) q[14];
rz(-2.0663194270785943) q[14];
ry(0.30810610178460873) q[15];
rz(2.2415108422598427) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.008186950214908423) q[0];
rz(-1.4721504215931418) q[0];
ry(1.5697176390644412) q[1];
rz(-2.3055398985919147) q[1];
ry(3.0002914527243227) q[2];
rz(-3.0239556129571916) q[2];
ry(-3.0238179958209352) q[3];
rz(-1.4971125535178311) q[3];
ry(-0.0002959598039946857) q[4];
rz(-0.039149657592382425) q[4];
ry(-1.5704599870860647) q[5];
rz(-1.7581661904655554) q[5];
ry(-1.5705261154513874) q[6];
rz(1.5706400559106095) q[6];
ry(1.5705393207322134) q[7];
rz(-1.570789669728509) q[7];
ry(-3.1414398175604994) q[8];
rz(-1.5051280640974216) q[8];
ry(-0.00014283164260575631) q[9];
rz(2.622035102746161) q[9];
ry(1.5704930588674733) q[10];
rz(3.141505729687194) q[10];
ry(1.73471515225066) q[11];
rz(-3.141409129083029) q[11];
ry(-1.5709984436501667) q[12];
rz(1.6049931295640878) q[12];
ry(-3.1414885811439417) q[13];
rz(1.5576371465048018) q[13];
ry(0.01404730272024235) q[14];
rz(0.9043716297758575) q[14];
ry(-0.013219518393389171) q[15];
rz(-2.7911662201219225) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.130526773599415) q[0];
rz(1.7001133632465875) q[0];
ry(3.123538437711317) q[1];
rz(0.8776147133478354) q[1];
ry(1.5713507455415565) q[2];
rz(0.20605996847722652) q[2];
ry(2.939668313379255) q[3];
rz(-2.478684996336324) q[3];
ry(0.008515162894822126) q[4];
rz(-1.9657827560685064) q[4];
ry(-1.5705973415878773) q[5];
rz(-1.6583306644206495) q[5];
ry(1.570479254112038) q[6];
rz(1.6656861424616185) q[6];
ry(-1.571124427404957) q[7];
rz(0.0016517048530539926) q[7];
ry(1.5709761678670233) q[8];
rz(1.5716699806082701) q[8];
ry(1.5707516589040234) q[9];
rz(-2.003831631815551) q[9];
ry(-1.8178853800041954) q[10];
rz(0.0034331263447827986) q[10];
ry(1.5708324848656599) q[11];
rz(-1.7440128572755986) q[11];
ry(-1.5707233013408795) q[12];
rz(1.406435200354522) q[12];
ry(3.138879969712329) q[13];
rz(-1.1916260583800709) q[13];
ry(0.0027834333970195857) q[14];
rz(-2.6170937405044934) q[14];
ry(-3.1411467661077586) q[15];
rz(0.5512751824019633) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5681999944731109) q[0];
rz(0.0014783814694858677) q[0];
ry(-3.1403242164548777) q[1];
rz(-2.946394042248625) q[1];
ry(3.1415873459598602) q[2];
rz(1.740073987211547) q[2];
ry(0.0005709389240625252) q[3];
rz(-0.6672629243625661) q[3];
ry(3.1412357603385246) q[4];
rz(-1.7493469074749255) q[4];
ry(-3.1403440989256635) q[5];
rz(-0.7217039646810344) q[5];
ry(-3.1405190852289744) q[6];
rz(0.26599673170300825) q[6];
ry(1.572219960221224) q[7];
rz(-2.3694001732545042) q[7];
ry(1.5694025752932068) q[8];
rz(-1.9214779650595881) q[8];
ry(-3.1397985209922767) q[9];
rz(-2.0030833108154065) q[9];
ry(-0.09149095158756587) q[10];
rz(0.19379118280920796) q[10];
ry(0.0019589065305263276) q[11];
rz(-0.24961989593623723) q[11];
ry(-0.0012652309914134021) q[12];
rz(-2.975217058428868) q[12];
ry(1.5709734099117953) q[13];
rz(-0.2556337961143473) q[13];
ry(-0.00023446803931381766) q[14];
rz(0.9619179773050579) q[14];
ry(-3.1414484750456104) q[15];
rz(1.9550813355965477) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5676724064636707) q[0];
rz(1.6017725922199333) q[0];
ry(3.140908369729308) q[1];
rz(1.4988917377217739) q[1];
ry(3.1415348793235607) q[2];
rz(-1.7351053503695875) q[2];
ry(-1.5574045041393192) q[3];
rz(0.035342051927423235) q[3];
ry(-0.00020633911777039385) q[4];
rz(-3.121124453554356) q[4];
ry(0.0007287518445231198) q[5];
rz(-0.7158204084166977) q[5];
ry(-3.1415158440511797) q[6];
rz(0.17109885594414023) q[6];
ry(-5.043008548710758e-05) q[7];
rz(0.7983940659281017) q[7];
ry(3.1414868095473314) q[8];
rz(-1.348536143702303) q[8];
ry(-0.18285406113087413) q[9];
rz(1.570212717610456) q[9];
ry(3.1415879358896404) q[10];
rz(2.796150626040261) q[10];
ry(3.990441847930525e-07) q[11];
rz(-2.7191264173607204) q[11];
ry(1.57043601192985) q[12];
rz(2.005021525647284) q[12];
ry(3.140514780041831) q[13];
rz(1.3151738337171668) q[13];
ry(-3.140701185174331) q[14];
rz(0.7130149493633079) q[14];
ry(0.0005421069286847386) q[15];
rz(-1.8658912733827284) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.7210583235622483) q[0];
rz(-2.930067236132651) q[0];
ry(1.5603774832027293) q[1];
rz(-3.1386815672703348) q[1];
ry(-0.00012448518371523673) q[2];
rz(1.493944623225703) q[2];
ry(-1.571231728171715) q[3];
rz(1.5704378900734293) q[3];
ry(3.141466405845955) q[4];
rz(1.167979503769172) q[4];
ry(0.0002207557113108364) q[5];
rz(1.3505187771627378) q[5];
ry(-1.5696387752895469) q[6];
rz(-0.0042311430937733065) q[6];
ry(1.5723267661616163) q[7];
rz(0.0018818462172509596) q[7];
ry(0.0015432218356466868) q[8];
rz(1.983385288616198) q[8];
ry(1.5722763003202918) q[9];
rz(1.5587616319361568) q[9];
ry(3.1414145650144576) q[10];
rz(1.0281928250203167) q[10];
ry(3.0583585089800094) q[11];
rz(-0.00045309120605274694) q[11];
ry(3.1387309123832465) q[12];
rz(-1.0584693758237744) q[12];
ry(1.5695143806099152) q[13];
rz(1.6077175752076616) q[13];
ry(1.5702817230251713) q[14];
rz(0.19983601218787017) q[14];
ry(-2.7933545919012207e-05) q[15];
rz(-0.4225939263636089) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5692530625006829) q[0];
rz(0.17027813427299243) q[0];
ry(-1.569287430074298) q[1];
rz(-2.015570302278925) q[1];
ry(-3.4428354735105415e-05) q[2];
rz(-1.3703479944957526) q[2];
ry(1.5681830213281747) q[3];
rz(0.6905968398249391) q[3];
ry(3.1383800113030356) q[4];
rz(1.1429400319165472) q[4];
ry(-1.570324820568298) q[5];
rz(0.01465745034330687) q[5];
ry(-1.5717086887439708) q[6];
rz(3.1363487855681904) q[6];
ry(1.5709249124063063) q[7];
rz(0.9835863466700906) q[7];
ry(3.1397468183321786) q[8];
rz(2.5922027091050612) q[8];
ry(1.5807865642713286) q[9];
rz(-2.878888992037448) q[9];
ry(1.5707908127416284) q[10];
rz(-1.5623480961307443) q[10];
ry(1.5718321544509883) q[11];
rz(1.4525769403181659) q[11];
ry(1.5708252216947343) q[12];
rz(-1.3633545269132312) q[12];
ry(1.570695681130375) q[13];
rz(3.1414915797557947) q[13];
ry(-3.1385385583956507) q[14];
rz(-2.3789299374837856) q[14];
ry(3.141111174334557) q[15];
rz(0.002457670591913987) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.1377413379197) q[0];
rz(0.17104080987285464) q[0];
ry(-3.141524062008419) q[1];
rz(-2.0159198399519065) q[1];
ry(-2.7464949287795792) q[2];
rz(-0.09801281449038424) q[2];
ry(0.00028114321694339145) q[3];
rz(2.561379234358974) q[3];
ry(2.589081992265818e-05) q[4];
rz(-1.7284979741503803) q[4];
ry(0.0004020723811777316) q[5];
rz(-1.5851878564887694) q[5];
ry(-0.0730711623535214) q[6];
rz(-1.7396546636017114) q[6];
ry(0.00014205403859908329) q[7];
rz(1.7924458476902634) q[7];
ry(-0.00014718547449937662) q[8];
rz(3.1107613986795073) q[8];
ry(8.975338677281807e-05) q[9];
rz(2.878425793260333) q[9];
ry(3.140769208860684) q[10];
rz(-1.5623648902464664) q[10];
ry(6.991688702172284e-05) q[11];
rz(-1.5614190023260262) q[11];
ry(0.00019799550979548997) q[12];
rz(2.9336422760866614) q[12];
ry(1.570776306599842) q[13];
rz(-0.006218375805605022) q[13];
ry(3.1413757593622327) q[14];
rz(0.5625784624519108) q[14];
ry(-1.5710577553858487) q[15];
rz(0.37793788586739285) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5697284061883217) q[0];
rz(-0.5419841856823339) q[0];
ry(-1.5726075350930735) q[1];
rz(-0.43069020744556064) q[1];
ry(-3.141544329473546) q[2];
rz(-0.6393626489046802) q[2];
ry(-3.138792171740096) q[3];
rz(1.2512333348486473) q[3];
ry(-0.0008897944159933503) q[4];
rz(2.664412218455484) q[4];
ry(-1.5707849697108731) q[5];
rz(-0.4295345415833047) q[5];
ry(3.1376318799712712) q[6];
rz(2.422175043655724) q[6];
ry(-3.141183015160554) q[7];
rz(-0.7955980705094505) q[7];
ry(-1.572333446732153) q[8];
rz(1.0245046729633227) q[8];
ry(1.5805579293203547) q[9];
rz(-0.4190127016656505) q[9];
ry(-1.5707228757439697) q[10];
rz(-2.111188269251838) q[10];
ry(3.141569327617056) q[11];
rz(1.0339168036607287) q[11];
ry(-1.5707311457450093) q[12];
rz(2.600493661224111) q[12];
ry(3.134257231979601) q[13];
rz(1.1332695644505608) q[13];
ry(-1.5708765044103856) q[14];
rz(-2.1101518746799948) q[14];
ry(-3.1410903924230653) q[15];
rz(3.0886045672818376) q[15];