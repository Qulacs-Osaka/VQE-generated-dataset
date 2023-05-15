OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.5555257924545263) q[0];
rz(1.058779874517611) q[0];
ry(-1.5806801225387095) q[1];
rz(0.12371184564079851) q[1];
ry(-1.9121621141546692) q[2];
rz(1.0047541006787046) q[2];
ry(-1.5762213186812692) q[3];
rz(0.005602982807277891) q[3];
ry(1.6381580980858388) q[4];
rz(0.3375259100540306) q[4];
ry(1.6235496704646772) q[5];
rz(-0.19521057322536442) q[5];
ry(-1.5629357181557302) q[6];
rz(-0.6400593111860219) q[6];
ry(1.570142985031043) q[7];
rz(1.1952993592167376) q[7];
ry(3.130256972245323) q[8];
rz(3.085978687690852) q[8];
ry(1.5795237857655087) q[9];
rz(-3.141547086954984) q[9];
ry(-3.1390856308995687) q[10];
rz(2.9528922271404525) q[10];
ry(-0.00020918483898087109) q[11];
rz(-1.657561463700021) q[11];
ry(-0.6418684267497587) q[12];
rz(-2.757571597024657) q[12];
ry(-2.4162124072700166) q[13];
rz(0.5095206123155656) q[13];
ry(-3.1397054951405057) q[14];
rz(1.6506644137155084) q[14];
ry(-0.7613331587376004) q[15];
rz(-2.7791009933172486) q[15];
ry(2.01744700491424) q[16];
rz(0.7034891339698318) q[16];
ry(-1.5487702381202757) q[17];
rz(0.06744146162322058) q[17];
ry(1.5939615069044597) q[18];
rz(0.2892738399267223) q[18];
ry(-2.0150833488478455) q[19];
rz(2.80272056646769) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.24846441292295365) q[0];
rz(0.3760066267137516) q[0];
ry(0.1251301782129836) q[1];
rz(-2.6821563586286365) q[1];
ry(0.9157608242437583) q[2];
rz(-2.146098031663979) q[2];
ry(-0.9169097013186134) q[3];
rz(-1.5665528598321588) q[3];
ry(3.1411837786481738) q[4];
rz(2.4457621803190177) q[4];
ry(-3.141411357011362) q[5];
rz(-1.7690590918845963) q[5];
ry(-0.0047631167600696855) q[6];
rz(-0.9450857364942167) q[6];
ry(-0.001343959883544521) q[7];
rz(-2.7664752341349397) q[7];
ry(-0.012464538508563376) q[8];
rz(-1.6707761249266453) q[8];
ry(2.453714898736761) q[9];
rz(-1.6398399725541364) q[9];
ry(-0.5186767304518272) q[10];
rz(-2.03001572945284) q[10];
ry(-0.0020165380821204337) q[11];
rz(0.1487322056081627) q[11];
ry(2.375241516724479) q[12];
rz(1.0026408778079068) q[12];
ry(-2.181429546284756) q[13];
rz(-0.8806230193490769) q[13];
ry(0.0025136810393240334) q[14];
rz(-2.222221396747832) q[14];
ry(0.00026378595794406267) q[15];
rz(0.28682844957500453) q[15];
ry(-0.00026771098958191706) q[16];
rz(1.1466770865805587) q[16];
ry(-3.1385014432369855) q[17];
rz(1.6070484773633622) q[17];
ry(3.065603175208889) q[18];
rz(1.0077065082473637) q[18];
ry(-3.0703869947813436) q[19];
rz(2.7839558010517544) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.5612567414605456) q[0];
rz(1.6361266722272552) q[0];
ry(-3.1398840851300798) q[1];
rz(0.5829135637926061) q[1];
ry(0.6009266442373764) q[2];
rz(-1.749265256097047) q[2];
ry(-1.3689016288692173) q[3];
rz(-1.9699665155253718) q[3];
ry(-3.084008688620088) q[4];
rz(-2.545638207515458) q[4];
ry(-1.449142555926312) q[5];
rz(-0.599600089902847) q[5];
ry(2.6772891120835984) q[6];
rz(1.866667420118984) q[6];
ry(1.106120088429333) q[7];
rz(2.977687296970557) q[7];
ry(1.5702850849416297) q[8];
rz(-1.5716590485696198) q[8];
ry(3.1221152879026532) q[9];
rz(-1.8132336266692128) q[9];
ry(-3.1410615973866407) q[10];
rz(-2.0459214496164293) q[10];
ry(3.141237321565818) q[11];
rz(-1.9239802353588278) q[11];
ry(-1.5813995910352643) q[12];
rz(1.5303022066304033) q[12];
ry(-1.5725476819617352) q[13];
rz(1.587720025300393) q[13];
ry(0.001481478953300197) q[14];
rz(-3.1193000639526174) q[14];
ry(-2.1376021209697615) q[15];
rz(0.9941248672315528) q[15];
ry(-1.083692323863944) q[16];
rz(-0.5168162432723529) q[16];
ry(-0.8693057895017624) q[17];
rz(-3.1188958476649504) q[17];
ry(-0.39559019685121033) q[18];
rz(0.37729939854067795) q[18];
ry(2.4084747498323162) q[19];
rz(2.7792507268808566) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5802142514417827) q[0];
rz(0.5435167284923274) q[0];
ry(1.5911086281960785) q[1];
rz(0.8085994898645037) q[1];
ry(1.798522752501788) q[2];
rz(1.429914037014086) q[2];
ry(-0.0007580560354420104) q[3];
rz(-1.1690274130067655) q[3];
ry(3.1412551757419007) q[4];
rz(-0.26378614786063626) q[4];
ry(-3.141392581218934) q[5];
rz(0.9943741059737888) q[5];
ry(0.21947500468762274) q[6];
rz(1.9188560219117683) q[6];
ry(3.0978996776140595) q[7];
rz(1.0668928882522817) q[7];
ry(-1.7082190503749395) q[8];
rz(0.9692050997885909) q[8];
ry(0.0015715610959158195) q[9];
rz(1.347585271525052) q[9];
ry(-1.5709553513255843) q[10];
rz(3.139853441169195) q[10];
ry(-0.10086662926836798) q[11];
rz(1.5409253706235733) q[11];
ry(-2.918128606332366) q[12];
rz(-1.485665655587711) q[12];
ry(0.6463417910071844) q[13];
rz(-2.715182062641199) q[13];
ry(3.141535791739091) q[14];
rz(-1.9022790502196951) q[14];
ry(-0.0004417943409599657) q[15];
rz(1.7958044715989105) q[15];
ry(-1.5689457349517053) q[16];
rz(1.2823816945639739) q[16];
ry(1.5725535300321465) q[17];
rz(1.586660140203642) q[17];
ry(0.6437252633632138) q[18];
rz(2.3277883712224305) q[18];
ry(1.9441884924008437) q[19];
rz(-2.460614038817308) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.048924091274257) q[0];
rz(-2.2214816413503855) q[0];
ry(-0.002933952482759222) q[1];
rz(-2.7122994710543438) q[1];
ry(1.457237062333069) q[2];
rz(0.32672237072180105) q[2];
ry(-1.5418048944927776) q[3];
rz(1.572370391281737) q[3];
ry(0.015726100459000206) q[4];
rz(1.1575000320533388) q[4];
ry(0.01003781131999124) q[5];
rz(0.8671558937014733) q[5];
ry(3.126765216236409) q[6];
rz(1.1452200706725133) q[6];
ry(-0.0004251245716959673) q[7];
rz(-1.2320240056790988) q[7];
ry(3.140611432677228) q[8];
rz(1.9646256669054027) q[8];
ry(-0.0006033864997130323) q[9];
rz(0.3962557067756576) q[9];
ry(1.5711832341234526) q[10];
rz(1.4221258148812161) q[10];
ry(1.5709208037117688) q[11];
rz(1.448085383559406) q[11];
ry(1.5647908598902303) q[12];
rz(-2.0193633753774143) q[12];
ry(-3.141221082832798) q[13];
rz(-1.5047803196793117) q[13];
ry(-1.495171465978184) q[14];
rz(3.072263074477156) q[14];
ry(-1.5192196254452293) q[15];
rz(-3.112556608640043) q[15];
ry(6.801325213142888e-05) q[16];
rz(-3.008374558264985) q[16];
ry(3.139722313689307) q[17];
rz(-1.809174253718484) q[17];
ry(-0.6738664271306618) q[18];
rz(0.5224102551678538) q[18];
ry(-2.698970436602961) q[19];
rz(-0.27536059030931276) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.044781198170306194) q[0];
rz(1.2621785196117874) q[0];
ry(1.4842694366947216) q[1];
rz(1.348789989768362) q[1];
ry(3.1237064053811867) q[2];
rz(1.4648576886729028) q[2];
ry(1.9951445980069176) q[3];
rz(2.520949516213569) q[3];
ry(-3.1408040012691774) q[4];
rz(2.403020322351047) q[4];
ry(-0.000660312319885271) q[5];
rz(-2.829905586135553) q[5];
ry(0.07123531291805307) q[6];
rz(0.9955192073305161) q[6];
ry(1.615667963307673) q[7];
rz(-0.18325083104619003) q[7];
ry(1.5708955128777566) q[8];
rz(-3.141450253625612) q[8];
ry(-1.571129865364826) q[9];
rz(-0.0002989558464676634) q[9];
ry(0.00015281254078857587) q[10];
rz(2.3165183191788814) q[10];
ry(0.0006856820581804257) q[11];
rz(-0.4014875977396386) q[11];
ry(-3.1414695124425216) q[12];
rz(2.646893380269147) q[12];
ry(0.00018745113507521942) q[13];
rz(2.0574572489800254) q[13];
ry(2.9219774359764066) q[14];
rz(-1.500750555596826) q[14];
ry(-2.853357029505211) q[15];
rz(1.696894997980865) q[15];
ry(0.02961980631923236) q[16];
rz(1.6705514761706484) q[16];
ry(-0.01838523277092996) q[17];
rz(-0.6978374961203041) q[17];
ry(-1.0082289576634116) q[18];
rz(-2.817351429684056) q[18];
ry(-0.442329532399536) q[19];
rz(2.6424987682363006) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1349211767382292) q[0];
rz(2.7972472412898495) q[0];
ry(1.0158174490773852) q[1];
rz(1.854002214821211) q[1];
ry(0.0001180933505180448) q[2];
rz(-0.9697570775968734) q[2];
ry(3.112811149046677) q[3];
rz(0.958029752634455) q[3];
ry(1.5713368603656577) q[4];
rz(-0.6109085999627962) q[4];
ry(-3.139612036117018) q[5];
rz(2.4204449163670727) q[5];
ry(3.1376559767292296) q[6];
rz(3.057690489433871) q[6];
ry(1.581564295395519) q[7];
rz(-1.5979383213395053) q[7];
ry(-1.5710571741450887) q[8];
rz(-1.5637848256083118) q[8];
ry(1.5708237687789612) q[9];
rz(0.9095971579127697) q[9];
ry(-3.141204341227695) q[10];
rz(-2.5462102129128463) q[10];
ry(3.124822907142969) q[11];
rz(2.617009409671161) q[11];
ry(2.9887792596612917) q[12];
rz(-0.604402545360994) q[12];
ry(0.0007401444733332596) q[13];
rz(-2.5056321337941405) q[13];
ry(-2.637025046273433) q[14];
rz(-1.411663679998578) q[14];
ry(-0.5037871439446819) q[15];
rz(-1.6803119559575357) q[15];
ry(0.0012807768443279916) q[16];
rz(2.1149832539615074) q[16];
ry(3.1400537595405695) q[17];
rz(1.3370819615487735) q[17];
ry(-0.43014016349034634) q[18];
rz(2.6368794781718856) q[18];
ry(1.2677852167677854) q[19];
rz(-1.7958266931634028) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.4400910373488454) q[0];
rz(-0.24334405103899825) q[0];
ry(-1.6848675220779077) q[1];
rz(-0.654147776263236) q[1];
ry(-1.5718579529028727) q[2];
rz(-1.568500258893809) q[2];
ry(-1.5656039463530949) q[3];
rz(2.2735757859552175) q[3];
ry(3.136419340978663) q[4];
rz(1.1355826514631122) q[4];
ry(-0.0002933527739718751) q[5];
rz(-0.24874798168189677) q[5];
ry(-1.63949326406712) q[6];
rz(0.00018317175399052522) q[6];
ry(-0.07673555729997616) q[7];
rz(1.8583297654076718) q[7];
ry(-1.5691528956916772) q[8];
rz(-1.562598097848532) q[8];
ry(-0.001498169229035895) q[9];
rz(-0.23021774048731544) q[9];
ry(-1.570775371516334) q[10];
rz(0.046232763211708765) q[10];
ry(1.5703321611223258) q[11];
rz(-1.6912589011612356) q[11];
ry(3.141483852556924) q[12];
rz(-0.5581685440733789) q[12];
ry(-0.0006286956458532553) q[13];
rz(1.5572771498011944) q[13];
ry(1.7679661035633885) q[14];
rz(-0.8385251322573126) q[14];
ry(-1.241459653370975) q[15];
rz(-0.001274114810662752) q[15];
ry(-3.103475494621943) q[16];
rz(2.8298584641819438) q[16];
ry(-3.1211552313661644) q[17];
rz(2.1494895882418517) q[17];
ry(1.4852760738051067) q[18];
rz(0.6746331377992743) q[18];
ry(-1.4751834760410734) q[19];
rz(-0.38523465704543464) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.003688857191017192) q[0];
rz(0.7938252310770818) q[0];
ry(-3.1384742066365887) q[1];
rz(-0.2867294134634797) q[1];
ry(1.5785078477126244) q[2];
rz(-0.04068246488319649) q[2];
ry(3.140748500218436) q[3];
rz(2.2723220748167754) q[3];
ry(3.141549139949308) q[4];
rz(1.7495410560879472) q[4];
ry(-0.00024325720258655537) q[5];
rz(-0.31863883431859996) q[5];
ry(1.5714799167891318) q[6];
rz(0.08632775160703224) q[6];
ry(3.141125707568738) q[7];
rz(1.8307619070536711) q[7];
ry(-0.0003737604855621393) q[8];
rz(0.06846738237810265) q[8];
ry(-0.0001037833266321897) q[9];
rz(-2.3227583384744896) q[9];
ry(-1.6221466410342908) q[10];
rz(-2.413606295032821) q[10];
ry(-3.0786040878446115) q[11];
rz(-1.6665457251507432) q[11];
ry(-1.5693777580850765) q[12];
rz(0.8082602441976543) q[12];
ry(-3.141231277113691) q[13];
rz(-1.0952501407638018) q[13];
ry(2.760871139243158) q[14];
rz(2.2412691522804042) q[14];
ry(1.5563305549438577) q[15];
rz(0.6473864154596644) q[15];
ry(-3.140560861542451) q[16];
rz(2.330126134632494) q[16];
ry(-1.561298885725753) q[17];
rz(-1.6019016527299819) q[17];
ry(1.3525898250762103) q[18];
rz(-2.5137500112053566) q[18];
ry(0.5693464750593412) q[19];
rz(0.9389648702300945) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1400794243349437) q[0];
rz(1.9227847593925196) q[0];
ry(3.1185950408533767) q[1];
rz(1.5921380673438543) q[1];
ry(-0.013416733446985742) q[2];
rz(1.613529169772927) q[2];
ry(1.5704117196855543) q[3];
rz(-2.1575760051963186) q[3];
ry(-0.526731428347352) q[4];
rz(3.1407348137166715) q[4];
ry(3.1411873002125796) q[5];
rz(-2.5192973166357917) q[5];
ry(0.06953939946411136) q[6];
rz(-0.08613038108517325) q[6];
ry(-1.5713905729784108) q[7];
rz(-0.9252405082997432) q[7];
ry(3.1414491947846184) q[8];
rz(2.8799967266625583) q[8];
ry(0.017003817807126786) q[9];
rz(-1.4983329016768179) q[9];
ry(-3.1406376296079417) q[10];
rz(0.7295409196517668) q[10];
ry(-3.14126344198049) q[11];
rz(0.023883694056790687) q[11];
ry(-0.0001376146025755531) q[12];
rz(2.959227226454595) q[12];
ry(-3.141216408440992) q[13];
rz(-0.27210886409185653) q[13];
ry(-2.717981134506295) q[14];
rz(-1.4328004467456141) q[14];
ry(0.0001484239285527892) q[15];
rz(0.253011194788618) q[15];
ry(3.140127036393905) q[16];
rz(2.9419916158058426) q[16];
ry(-0.001133026665321779) q[17];
rz(-1.5424147791754583) q[17];
ry(3.1014115177430392) q[18];
rz(1.4477593680881613) q[18];
ry(-1.565399839943534) q[19];
rz(-2.9638266125695285) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.7759788682599194) q[0];
rz(-3.0991428241054737) q[0];
ry(1.5300624399922276) q[1];
rz(-1.280403336539659) q[1];
ry(-1.5625091975089742) q[2];
rz(-0.00378083317407906) q[2];
ry(-1.5701593578876993) q[3];
rz(-1.5693500444059525) q[3];
ry(-1.5592998704248329) q[4];
rz(-1.483498390676873) q[4];
ry(1.5703975911555692) q[5];
rz(1.5708923137742428) q[5];
ry(1.6154920105871682) q[6];
rz(2.9988910982167996) q[6];
ry(-3.1397861584785702) q[7];
rz(-2.6110563135431555) q[7];
ry(-0.0017852015555801515) q[8];
rz(-1.625608016593117) q[8];
ry(1.5716372038859108) q[9];
rz(0.01591681170964983) q[9];
ry(-1.4987885905108946) q[10];
rz(3.096944622160144) q[10];
ry(-1.5780351886625104) q[11];
rz(3.0772092064385026) q[11];
ry(-3.140722050885016) q[12];
rz(2.0763863771179274) q[12];
ry(1.5718871653864466) q[13];
rz(-1.071795060445728) q[13];
ry(2.892558575068419) q[14];
rz(1.4694031539805816) q[14];
ry(-2.252304336722643) q[15];
rz(0.5540938099349884) q[15];
ry(-0.10041570288054925) q[16];
rz(-0.21171462499862773) q[16];
ry(-1.5489772868922431) q[17];
rz(-0.37286599881256405) q[17];
ry(1.5641361437866754) q[18];
rz(1.8180490486749328) q[18];
ry(1.7345949072268647) q[19];
rz(-1.5391557380692606) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5714963171554748) q[0];
rz(3.1383610947958185) q[0];
ry(0.17354185405483807) q[1];
rz(-1.6602193541795138) q[1];
ry(-1.510360108213014) q[2];
rz(-1.5694466546879866) q[2];
ry(1.5694872028546207) q[3];
rz(1.5678114254230595) q[3];
ry(3.1407068570825665) q[4];
rz(0.08644820597938097) q[4];
ry(-2.0752118263229695) q[5];
rz(-0.0001866696609873486) q[5];
ry(3.1411773246854344) q[6];
rz(1.42804565698519) q[6];
ry(-0.0004549283437022983) q[7];
rz(0.11507541943854337) q[7];
ry(-3.1415917355349396) q[8];
rz(-1.9639680688313907) q[8];
ry(3.106137701713863e-06) q[9];
rz(0.0736743220445303) q[9];
ry(2.725530914512713) q[10];
rz(-3.1398683944662196) q[10];
ry(-0.020722157999184268) q[11];
rz(0.11961502156014503) q[11];
ry(-0.0002348774564077633) q[12];
rz(-3.021990495198501) q[12];
ry(3.141248811136815) q[13];
rz(-2.6429486729377984) q[13];
ry(0.0012822791537425715) q[14];
rz(1.813751383677633) q[14];
ry(-0.00034215454225447006) q[15];
rz(1.446629828697219) q[15];
ry(3.128343761555435) q[16];
rz(-1.8324490062979892) q[16];
ry(0.0009554088887665611) q[17];
rz(0.3709842594721397) q[17];
ry(0.00018767113529083446) q[18];
rz(2.8870685134025718) q[18];
ry(1.5694961996585646) q[19];
rz(-0.005843157124299837) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5540639957917222) q[0];
rz(-0.003254966143191033) q[0];
ry(-1.5700292584010473) q[1];
rz(-0.0028132972524272507) q[1];
ry(-1.569981366865643) q[2];
rz(1.5619426209730243) q[2];
ry(-1.5708197474373098) q[3];
rz(0.0008738325515704101) q[3];
ry(-1.5690894979111054) q[4];
rz(-2.214340191476566) q[4];
ry(1.5709801197043112) q[5];
rz(3.1235846341125355) q[5];
ry(1.5708302073925646) q[6];
rz(0.9918475114649329) q[6];
ry(1.5709914273550254) q[7];
rz(-2.030184362438464) q[7];
ry(1.573566392908863) q[8];
rz(-0.046875224358148955) q[8];
ry(0.0009691724088343179) q[9];
rz(-0.8300957484917411) q[9];
ry(-1.592150086911711) q[10];
rz(-1.500445125258866) q[10];
ry(3.1412670247030072) q[11];
rz(1.6894711347883704) q[11];
ry(-1.570985605562573) q[12];
rz(-2.339519543200327) q[12];
ry(-1.580987542486822) q[13];
rz(1.571815842280649) q[13];
ry(1.3240426904719413) q[14];
rz(2.0532290996981155) q[14];
ry(2.044734127281775) q[15];
rz(-0.5497886103582408) q[15];
ry(1.5712702905964395) q[16];
rz(-2.461694193493383) q[16];
ry(-1.5960307181153004) q[17];
rz(-2.979375643617008) q[17];
ry(-1.237185343742154) q[18];
rz(1.57243547844262) q[18];
ry(1.4032277980977519) q[19];
rz(0.0024632727261166554) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.0506219707166089) q[0];
rz(1.748270861320938) q[0];
ry(3.103261850575148) q[1];
rz(-1.571071071401277) q[1];
ry(1.5699443215620548) q[2];
rz(3.110558218157104) q[2];
ry(1.5827108438825206) q[3];
rz(-3.1235936596399685) q[3];
ry(-3.138225898705001) q[4];
rz(2.8879115825696653) q[4];
ry(1.571037664048906) q[5];
rz(1.529831757046526) q[5];
ry(-8.728802623370058e-05) q[6];
rz(1.7123009760564) q[6];
ry(-0.0002970645564941421) q[7];
rz(0.9246455673412166) q[7];
ry(3.1410461603783917) q[8];
rz(-1.61441455371295) q[8];
ry(-0.00014077121268272358) q[9];
rz(2.3099871764554334) q[9];
ry(3.1413946712283787) q[10];
rz(1.6874728574364202) q[10];
ry(1.5695789725286178) q[11];
rz(-1.5794396013113299) q[11];
ry(3.140701630843897) q[12];
rz(2.38635001503791) q[12];
ry(-1.5704083674197133) q[13];
rz(3.0829996777781226) q[13];
ry(-0.0005141194427117649) q[14];
rz(-0.2615965017515487) q[14];
ry(-3.1409794571588874) q[15];
rz(0.2341537672643179) q[15];
ry(0.0002317471849294143) q[16];
rz(2.4662375145988062) q[16];
ry(-0.0026451217126392404) q[17];
rz(-1.6677717080941394) q[17];
ry(1.5904250142365237) q[18];
rz(1.8520890406994164) q[18];
ry(-1.57726305615568) q[19];
rz(-1.975571450650353) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.572257886617824) q[0];
rz(1.5570662743029338) q[0];
ry(1.5687779511066786) q[1];
rz(1.833889311044289) q[1];
ry(3.0264448589253337) q[2];
rz(1.5619288574469625) q[2];
ry(6.642632180131613e-05) q[3];
rz(-1.6356480595304033) q[3];
ry(-3.137389297710582) q[4];
rz(-2.7525332097865434) q[4];
ry(-3.1413781479531426) q[5];
rz(0.22555298098066853) q[5];
ry(-0.0002638300900796864) q[6];
rz(-1.128695507712175) q[6];
ry(3.141558800462983) q[7];
rz(-0.36466921452638434) q[7];
ry(-1.5388466896995512) q[8];
rz(-1.5952945801395542) q[8];
ry(1.5706622271519513) q[9];
rz(0.7886259406631286) q[9];
ry(3.1415517676559874) q[10];
rz(-3.095797685481444) q[10];
ry(-3.1292711342327872) q[11];
rz(-2.712432884412485) q[11];
ry(-0.0005716126264345068) q[12];
rz(-2.5831007120701046) q[12];
ry(-2.821864673485663e-06) q[13];
rz(-3.082246027577551) q[13];
ry(1.5606455496283345) q[14];
rz(3.1415315001480137) q[14];
ry(-1.5708543372921069) q[15];
rz(-1.5237223269664657) q[15];
ry(2.1019230867801175) q[16];
rz(3.028356202818004) q[16];
ry(-1.5705090572856548) q[17];
rz(-1.5897480682039307) q[17];
ry(-3.1343783514484254) q[18];
rz(-2.244226151930195) q[18];
ry(2.784092688394866) q[19];
rz(-1.9544129243216215) q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.6042891695571844) q[0];
rz(-0.7598160875264214) q[0];
ry(3.133168374522075) q[1];
rz(0.5490373458364857) q[1];
ry(-1.571188796689535) q[2];
rz(2.397098969767795) q[2];
ry(2.8100144488232237) q[3];
rz(-2.903726432908054) q[3];
ry(-1.5671991921577275) q[4];
rz(-2.3258603867034475) q[4];
ry(-0.018767167814101704) q[5];
rz(-1.5499825215889942) q[5];
ry(-1.5708187024039832) q[6];
rz(0.8196341890633612) q[6];
ry(-3.1415635609120036) q[7];
rz(-2.112200978089282) q[7];
ry(1.5704928334423853) q[8];
rz(0.8173155174392702) q[8];
ry(3.140832276174119) q[9];
rz(-2.0652423672569364) q[9];
ry(1.5704565338333918) q[10];
rz(-0.752589323638312) q[10];
ry(0.0036706098237702708) q[11];
rz(-0.150228652895646) q[11];
ry(-3.1409202384935866) q[12];
rz(2.96058154374261) q[12];
ry(-1.5710343577488635) q[13];
rz(-1.2832152424366692) q[13];
ry(-1.629700572105314) q[14];
rz(2.3895376637166788) q[14];
ry(-3.135706616933967) q[15];
rz(-2.8069406354813182) q[15];
ry(0.024772436392681563) q[16];
rz(-0.6321790039541996) q[16];
ry(-1.5733573494062223) q[17];
rz(-1.2873949192553624) q[17];
ry(3.1408209109785714) q[18];
rz(-1.7078360097224543) q[18];
ry(-1.5736852873788856) q[19];
rz(1.8543550981023118) q[19];