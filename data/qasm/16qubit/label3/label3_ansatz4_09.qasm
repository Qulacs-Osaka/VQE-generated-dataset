OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.57079275927655) q[0];
rz(-0.11737966431136738) q[0];
ry(1.5651484320270175) q[1];
rz(-2.489692021292263) q[1];
ry(-2.0048207872983648) q[2];
rz(1.5705267885110041) q[2];
ry(-3.050509237310655) q[3];
rz(-0.001005161696401245) q[3];
ry(1.5703838443833646) q[4];
rz(-1.564513876795969) q[4];
ry(0.00017834664697335967) q[5];
rz(-2.7322832433384385) q[5];
ry(-1.5707612937724735) q[6];
rz(-1.6907489930984436) q[6];
ry(1.5707959810064143) q[7];
rz(1.7593143844697146) q[7];
ry(-0.017804107623813792) q[8];
rz(2.0638069745986005) q[8];
ry(-0.00014348608144931774) q[9];
rz(-1.7349580184075952) q[9];
ry(1.571188532198309) q[10];
rz(3.1415603348980903) q[10];
ry(1.5707594097181588) q[11];
rz(0.36112518178555103) q[11];
ry(1.573533692257933) q[12];
rz(0.6322070465829155) q[12];
ry(3.141253343368469) q[13];
rz(2.1333700611627586) q[13];
ry(1.5705920333918533) q[14];
rz(-1.5707355563165526) q[14];
ry(2.3113782646812793e-05) q[15];
rz(2.1681504580870694) q[15];
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
ry(0.00011551918191899825) q[0];
rz(0.11643097711771323) q[0];
ry(-3.141577470432235) q[1];
rz(2.2226903139905527) q[1];
ry(1.4894091341850577) q[2];
rz(-3.1412520683535883) q[2];
ry(1.5705379207976575) q[3];
rz(0.7816810073458622) q[3];
ry(0.06909970966545112) q[4];
rz(-1.548255950929243) q[4];
ry(-0.00207419470376727) q[5];
rz(0.9702365547382122) q[5];
ry(-0.0063328783349954065) q[6];
rz(2.7954143763816117) q[6];
ry(1.5711287606309066) q[7];
rz(1.6665391390694915) q[7];
ry(-3.1414290209851443) q[8];
rz(-0.43189752686008903) q[8];
ry(1.4610076640409917) q[9];
rz(1.5706531889249942) q[9];
ry(-1.4426973730439006) q[10];
rz(0.0010431780873507777) q[10];
ry(-1.4429707163587802) q[11];
rz(1.4880879178203674) q[11];
ry(-3.1415361183082675) q[12];
rz(1.9306047634776533) q[12];
ry(-2.8882454497427235) q[13];
rz(1.6600759838840804) q[13];
ry(-1.8419197377520127) q[14];
rz(0.00024203579164176317) q[14];
ry(1.570367916979425) q[15];
rz(1.1888920719334337) q[15];
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
ry(1.569976788508506) q[0];
rz(2.4934853290173256) q[0];
ry(-2.209749358608578) q[1];
rz(1.5707801261993009) q[1];
ry(-2.711835687820214) q[2];
rz(1.5677758996427382) q[2];
ry(-1.886757950020069) q[3];
rz(-0.2880758281621135) q[3];
ry(-3.1259874501102303) q[4];
rz(-1.5420883544669834) q[4];
ry(-3.1415575943928964) q[5];
rz(1.9361171910818313) q[5];
ry(-3.1415783031566304) q[6];
rz(-0.4660182119049097) q[6];
ry(3.1415812976388806) q[7];
rz(-3.0462270960076463) q[7];
ry(-2.4304768275783472e-05) q[8];
rz(-2.223945557500505) q[8];
ry(0.1667648498136141) q[9];
rz(-3.141431902141081) q[9];
ry(-1.5708253272793522) q[10];
rz(1.8313241114522247) q[10];
ry(0.00010113877460175334) q[11];
rz(-2.1814730475736424) q[11];
ry(1.5719006140698255) q[12];
rz(0.5319682430279449) q[12];
ry(-0.0011147702929577008) q[13];
rz(0.8380964042745697) q[13];
ry(-2.5937466020434647) q[14];
rz(-1.5702610085249835) q[14];
ry(1.5709718374635342) q[15];
rz(-0.22500463728615336) q[15];
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
ry(1.5685299899866445) q[0];
rz(-1.5708013980635087) q[0];
ry(1.570941693492701) q[1];
rz(1.8060922328287) q[1];
ry(1.5653504073680804) q[2];
rz(-1.5705977882825075) q[2];
ry(1.5763960214127402) q[3];
rz(-3.1415019760765377) q[3];
ry(1.5994516615139984) q[4];
rz(-1.5706870388390846) q[4];
ry(1.57025003540262) q[5];
rz(-0.009448096569422044) q[5];
ry(1.5771620213830408) q[6];
rz(-1.3936104486292007) q[6];
ry(-1.8219682803087052) q[7];
rz(1.570561217535243) q[7];
ry(-1.5709345646639408) q[8];
rz(0.054967898235152565) q[8];
ry(-1.5708155437629887) q[9];
rz(2.1650862717508548) q[9];
ry(-8.026722555647168e-05) q[10];
rz(1.3393764073058965) q[10];
ry(0.04471080850847464) q[11];
rz(0.5590147714315838) q[11];
ry(-0.09209117961607927) q[12];
rz(1.6937692700652196) q[12];
ry(3.141578175544348) q[13];
rz(0.42738569343574984) q[13];
ry(-1.5717495208558614) q[14];
rz(-1.158976481463112) q[14];
ry(0.0014279021199759627) q[15];
rz(1.798863120117905) q[15];
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
ry(-1.356307791556347) q[0];
rz(-1.5683813940588) q[0];
ry(-1.5718281333385837) q[1];
rz(-1.4635206372194673) q[1];
ry(-1.574115381888392) q[2];
rz(-0.29954962592152734) q[2];
ry(1.5718922753508746) q[3];
rz(-1.565684094694145) q[3];
ry(1.5712150339845357) q[4];
rz(-1.3230771405757418) q[4];
ry(1.5706431942089343) q[5];
rz(-1.5707653509957167) q[5];
ry(-0.00016127392053277845) q[6];
rz(-0.17866953697052512) q[6];
ry(0.1335185970242926) q[7];
rz(1.5718201512414856) q[7];
ry(-0.00023960980019310796) q[8];
rz(1.515008388752167) q[8];
ry(-3.1415892647351247) q[9];
rz(-2.114770226410225) q[9];
ry(0.0012397516308587343) q[10];
rz(-0.029017061637976228) q[10];
ry(1.5705861323303447) q[11];
rz(-2.3096376251627038) q[11];
ry(1.5702474734143657) q[12];
rz(-3.1413590436664096) q[12];
ry(0.00019540611339031324) q[13];
rz(-2.6414284499604066) q[13];
ry(0.000528324703876315) q[14];
rz(-0.41179211514510083) q[14];
ry(3.1394219473815332) q[15];
rz(-1.606511008595651) q[15];
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
ry(-1.5707980554496812) q[0];
rz(-0.5302053697018133) q[0];
ry(0.0002491952703437275) q[1];
rz(-0.10685897970972479) q[1];
ry(-3.477727732459357e-07) q[2];
rz(-2.8423618047277066) q[2];
ry(-3.122532374251101) q[3];
rz(0.008210147841538663) q[3];
ry(-1.5706981043714459) q[4];
rz(1.2818685045195517) q[4];
ry(-1.5643732795472616) q[5];
rz(-2.5302625634323084) q[5];
ry(-1.6516915754388914) q[6];
rz(-1.3561974192011015) q[6];
ry(1.5686047248709656) q[7];
rz(0.26269692993077337) q[7];
ry(1.570678223161976) q[8];
rz(-1.574237183565847) q[8];
ry(-3.141292017842313) q[9];
rz(-1.1588480744649805) q[9];
ry(1.5702506367718145) q[10];
rz(-1.5711639972886475) q[10];
ry(-7.988845650348729e-05) q[11];
rz(-2.115834603034987) q[11];
ry(-1.570784518080088) q[12];
rz(0.4711709402105553) q[12];
ry(1.5781222638178773) q[13];
rz(-1.571017592751808) q[13];
ry(-1.5706627871456114) q[14];
rz(1.5692256335384673) q[14];
ry(0.00043445811497822007) q[15];
rz(0.0389248284329532) q[15];
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
ry(-3.141165152818426) q[0];
rz(-0.5269825489628237) q[0];
ry(-1.5710812756829575) q[1];
rz(1.5336245707254592) q[1];
ry(1.5711738968941593) q[2];
rz(6.081564587390176e-05) q[2];
ry(1.601839660911354) q[3];
rz(0.1998757923689424) q[3];
ry(-7.204329827205067e-05) q[4];
rz(1.7826992972375182) q[4];
ry(-0.00013309025710661615) q[5];
rz(-0.44557171693905423) q[5];
ry(-3.141512154612542) q[6];
rz(-2.9251358866465105) q[6];
ry(3.912287566073031e-05) q[7];
rz(-0.2620806237278356) q[7];
ry(0.2500683355329185) q[8];
rz(0.9743844197198648) q[8];
ry(-0.024442981032739522) q[9];
rz(1.712154790745758) q[9];
ry(-2.8081643731588093) q[10];
rz(-3.1365955661120815) q[10];
ry(0.00024254061630580177) q[11];
rz(1.1886675208826172) q[11];
ry(0.005407974316267605) q[12];
rz(2.6704110400419725) q[12];
ry(1.5707050542270702) q[13];
rz(1.6126449467903718) q[13];
ry(-1.5709821582049788) q[14];
rz(0.10625480270826329) q[14];
ry(-1.570670094310172) q[15];
rz(1.5701875225928568) q[15];
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
ry(-2.494607639730335) q[0];
rz(-0.5982393400926744) q[0];
ry(-0.9208868440388712) q[1];
rz(-3.0506816834148305) q[1];
ry(-1.2892400373675175) q[2];
rz(1.5707579790509776) q[2];
ry(-3.141450260845847) q[3];
rz(-2.934687021309911) q[3];
ry(-3.1414231283248992) q[4];
rz(1.697194228800107) q[4];
ry(-1.7664968425812333e-05) q[5];
rz(-1.736191555767288) q[5];
ry(-1.5698228154446048) q[6];
rz(0.10945696256879785) q[6];
ry(-1.573236573390308) q[7];
rz(1.571688103950458) q[7];
ry(3.1414048884341588) q[8];
rz(-0.5962164684268146) q[8];
ry(3.1413246141194913) q[9];
rz(-2.97582367218547) q[9];
ry(0.059781648784925166) q[10];
rz(0.14642256171797197) q[10];
ry(0.009364283854806033) q[11];
rz(0.09536633387140263) q[11];
ry(-1.5707667212913101) q[12];
rz(0.07355754747760647) q[12];
ry(-1.5707362220446806) q[13];
rz(2.0652984193676462) q[13];
ry(0.029270735784137077) q[14];
rz(1.425588747013184) q[14];
ry(-1.5745918193603607) q[15];
rz(-3.1414411109476426) q[15];
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
ry(0.00025785409734606796) q[0];
rz(-0.9651864936178409) q[0];
ry(1.5708140894615914) q[1];
rz(-0.0028177541793308336) q[1];
ry(1.5702973834185574) q[2];
rz(3.1411697806854146) q[2];
ry(0.03203737626654658) q[3];
rz(0.7859292392645543) q[3];
ry(4.829194026441836e-05) q[4];
rz(2.938190216594508) q[4];
ry(0.06219572187057323) q[5];
rz(2.308947643066787) q[5];
ry(-1.5707852484495823) q[6];
rz(-1.1532221035892576e-05) q[6];
ry(-1.5708083782848714) q[7];
rz(3.141522586281137) q[7];
ry(2.943343336676903) q[8];
rz(2.4042142506050945) q[8];
ry(3.1381670860240325) q[9];
rz(1.3584757350308798) q[9];
ry(-0.0011892907102684092) q[10];
rz(-0.15093321014131966) q[10];
ry(1.5713071774565313) q[11];
rz(-0.5027937092013921) q[11];
ry(-1.5707189086192597) q[12];
rz(-1.5361267462806742) q[12];
ry(4.327949801652409e-05) q[13];
rz(-2.755525235634721) q[13];
ry(1.5705311860774587) q[14];
rz(1.5660586581586207) q[14];
ry(1.4861668311249954) q[15];
rz(-0.012956944914833821) q[15];
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
ry(-1.5704334023400452) q[0];
rz(-2.151116857392905) q[0];
ry(1.5641224885992564) q[1];
rz(-1.8273579845355774) q[1];
ry(-1.5701456098442481) q[2];
rz(-3.052867837289055) q[2];
ry(-0.0003938954815980722) q[3];
rz(0.771909283764128) q[3];
ry(-1.5710911667069427) q[4];
rz(-1.5711636150143236) q[4];
ry(-0.0002152916544043476) q[5];
rz(2.2686730510873643) q[5];
ry(1.5707918024328222) q[6];
rz(1.569925218022294) q[6];
ry(1.5707799692224471) q[7];
rz(3.139655486147271) q[7];
ry(1.5563432819831746e-05) q[8];
rz(2.7560773280236965) q[8];
ry(5.194351706089419e-06) q[9];
rz(1.1587162556774988) q[9];
ry(-0.2134299123981549) q[10];
rz(-1.0279943213164897) q[10];
ry(-0.00018012596283873905) q[11];
rz(-0.9139311758533255) q[11];
ry(3.141532702593395) q[12];
rz(2.126294982770773) q[12];
ry(3.140331768930227) q[13];
rz(-2.2607448019647762) q[13];
ry(1.5711363224999693) q[14];
rz(3.1415049143692126) q[14];
ry(0.11435530515693139) q[15];
rz(2.5740772591099836) q[15];
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
ry(3.133581787309641) q[0];
rz(2.513207263305673) q[0];
ry(-0.0008626884614885671) q[1];
rz(2.1693450568825514) q[1];
ry(-3.141572981829242) q[2];
rz(-0.5799449115031869) q[2];
ry(3.1097707811542814) q[3];
rz(-0.00602014233051662) q[3];
ry(2.8710402241853483) q[4];
rz(-0.0004139960645246284) q[4];
ry(-3.1414762445703386) q[5];
rz(-0.13439570244160082) q[5];
ry(-0.08605562114636167) q[6];
rz(2.8742783153532225) q[6];
ry(-1.4833089773673573) q[7];
rz(-1.701925249183872) q[7];
ry(2.907995383622164) q[8];
rz(-1.4174919242776876) q[8];
ry(-2.223032200014691) q[9];
rz(-1.6197420277751593) q[9];
ry(-1.8986745415135802) q[10];
rz(1.9657827801306234) q[10];
ry(2.985118472303765) q[11];
rz(-2.2045669947855107) q[11];
ry(-3.1410053896678396) q[12];
rz(2.787974912847557) q[12];
ry(1.5706887786029053) q[13];
rz(-1.1922480383557925) q[13];
ry(-1.610015151527823) q[14];
rz(-1.8273413150345017) q[14];
ry(-1.5738340004969018) q[15];
rz(1.568286945860268) q[15];
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
ry(0.007408427875688888) q[0];
rz(1.0498042396243177) q[0];
ry(3.1340163228054494) q[1];
rz(-1.8208457932206) q[1];
ry(-3.1413940896414436) q[2];
rz(-2.101323164483601) q[2];
ry(-1.5704516862408768) q[3];
rz(-3.139913365768492) q[3];
ry(1.570773000240486) q[4];
rz(3.0437716656211404) q[4];
ry(1.57083293991266) q[5];
rz(0.0005911636811450903) q[5];
ry(-3.1415705078532645) q[6];
rz(-0.2681347043548001) q[6];
ry(-3.141566766065617) q[7];
rz(1.1898926683772153) q[7];
ry(3.141583932554983) q[8];
rz(-1.8767680818681234) q[8];
ry(3.1415829699029696) q[9];
rz(-2.746809097576738) q[9];
ry(-1.817248424095817e-05) q[10];
rz(2.253343214132242) q[10];
ry(6.0172593014051756e-05) q[11];
rz(0.7897285281804707) q[11];
ry(-3.141507928601995) q[12];
rz(-0.8734781291202669) q[12];
ry(-2.7380246820488258e-05) q[13];
rz(0.4425454761164307) q[13];
ry(-3.140974321406577) q[14];
rz(2.8858387232501013) q[14];
ry(-1.5708493422757552) q[15];
rz(-3.138480760820937) q[15];
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
ry(-3.1415339500854884) q[0];
rz(2.2974795646791044) q[0];
ry(-0.0035786299300083115) q[1];
rz(-2.828014457001654) q[1];
ry(3.141571297283708) q[2];
rz(-0.13633541658955828) q[2];
ry(-1.567774775673394) q[3];
rz(-0.2783891718263183) q[3];
ry(-3.1411918984017406) q[4];
rz(-0.3723788729231927) q[4];
ry(1.5680914838001634) q[5];
rz(2.863422583937502) q[5];
ry(-1.5711925082810838) q[6];
rz(2.865011525502109) q[6];
ry(-0.0029481846564706773) q[7];
rz(3.1128568249914608) q[7];
ry(-1.7807200145598259) q[8];
rz(1.1916778757034043) q[8];
ry(-1.0566148723818436) q[9];
rz(2.097961389286307) q[9];
ry(0.6283745178234668) q[10];
rz(-1.2608377659928387) q[10];
ry(1.5482530948648527) q[11];
rz(1.4519780927972281) q[11];
ry(1.5739581546488792) q[12];
rz(-0.27593326061102974) q[12];
ry(0.004330627098746653) q[13];
rz(-1.0947950350381193) q[13];
ry(1.5739606871265885) q[14];
rz(-0.27585479715765704) q[14];
ry(-0.9920558997866602) q[15];
rz(-1.8462347096130012) q[15];