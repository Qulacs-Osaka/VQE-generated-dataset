OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.5665805059414692) q[0];
ry(-2.8283330086523186) q[1];
cx q[0],q[1];
ry(0.9020110678437459) q[0];
ry(2.0276663478820427) q[1];
cx q[0],q[1];
ry(1.517919951688915) q[2];
ry(2.693159119823768) q[3];
cx q[2],q[3];
ry(-1.9402632576380618) q[2];
ry(1.9974742843365978) q[3];
cx q[2],q[3];
ry(-1.2058440092223224) q[4];
ry(0.0055415142169331815) q[5];
cx q[4],q[5];
ry(-2.946682708903887) q[4];
ry(2.910594941272663) q[5];
cx q[4],q[5];
ry(-1.376910112984004) q[6];
ry(-0.7662646420627929) q[7];
cx q[6],q[7];
ry(0.2271904198505812) q[6];
ry(-1.2967738527985642) q[7];
cx q[6],q[7];
ry(-2.5440459576979637) q[8];
ry(-3.020553076128549) q[9];
cx q[8],q[9];
ry(-2.7961984879847592) q[8];
ry(-1.396912763879937) q[9];
cx q[8],q[9];
ry(-1.4910561964321811) q[10];
ry(-1.721164736060338) q[11];
cx q[10],q[11];
ry(-3.1068392242865674) q[10];
ry(-0.6655454994662909) q[11];
cx q[10],q[11];
ry(-0.48384380685961403) q[12];
ry(0.6627347010253573) q[13];
cx q[12],q[13];
ry(0.3686357022192883) q[12];
ry(2.152728455157603) q[13];
cx q[12],q[13];
ry(-2.868219145330201) q[14];
ry(0.5104445879619474) q[15];
cx q[14],q[15];
ry(2.2430292145654387) q[14];
ry(1.846289576155782) q[15];
cx q[14],q[15];
ry(-0.5835360575022229) q[16];
ry(-1.7144291010763029) q[17];
cx q[16],q[17];
ry(2.596305942114951) q[16];
ry(-1.0536752827352371) q[17];
cx q[16],q[17];
ry(1.9812776838651012) q[18];
ry(2.3166720027036516) q[19];
cx q[18],q[19];
ry(-2.9862843507139094) q[18];
ry(-2.8837482592583257) q[19];
cx q[18],q[19];
ry(1.6560034778463235) q[1];
ry(-2.1485432963856583) q[2];
cx q[1],q[2];
ry(-0.6113292112111536) q[1];
ry(-0.19187811555693557) q[2];
cx q[1],q[2];
ry(-3.055205263770931) q[3];
ry(-1.1714044155666095) q[4];
cx q[3],q[4];
ry(2.1721634727783177) q[3];
ry(-1.6453409585082055) q[4];
cx q[3],q[4];
ry(-0.15025376418263647) q[5];
ry(-2.9800872270084984) q[6];
cx q[5],q[6];
ry(1.5044425778627115) q[5];
ry(3.1207175126408013) q[6];
cx q[5],q[6];
ry(-2.080901053560204) q[7];
ry(2.9442197493606916) q[8];
cx q[7],q[8];
ry(-3.093033583267413) q[7];
ry(-3.1411949797954506) q[8];
cx q[7],q[8];
ry(0.718681843091499) q[9];
ry(2.6495116015299587) q[10];
cx q[9],q[10];
ry(-0.1937089764602602) q[9];
ry(3.140995947349108) q[10];
cx q[9],q[10];
ry(-1.7224114270981798) q[11];
ry(-1.2412872761165648) q[12];
cx q[11],q[12];
ry(0.0009617603592122492) q[11];
ry(0.0028393953301355523) q[12];
cx q[11],q[12];
ry(-1.7834726007783734) q[13];
ry(-0.2603826207347905) q[14];
cx q[13],q[14];
ry(-0.7985863528062591) q[13];
ry(0.44540815984325816) q[14];
cx q[13],q[14];
ry(2.1299882246386623) q[15];
ry(-0.08060743930440317) q[16];
cx q[15],q[16];
ry(-1.3249010034649562) q[15];
ry(-0.26155008532886637) q[16];
cx q[15],q[16];
ry(1.9011081836366763) q[17];
ry(-1.1306374180914314) q[18];
cx q[17],q[18];
ry(2.7840684811093515) q[17];
ry(-1.5461679979546863) q[18];
cx q[17],q[18];
ry(-0.9721162414289176) q[0];
ry(1.3918073622855438) q[1];
cx q[0],q[1];
ry(-0.1834533290389921) q[0];
ry(-1.3341367560457629) q[1];
cx q[0],q[1];
ry(-0.7850836510439178) q[2];
ry(-0.1607609706480444) q[3];
cx q[2],q[3];
ry(-3.141427945745933) q[2];
ry(-1.6117680920581519) q[3];
cx q[2],q[3];
ry(-1.1423475953975677) q[4];
ry(-3.0689296908404113) q[5];
cx q[4],q[5];
ry(0.8686092317001463) q[4];
ry(-0.12252809538587783) q[5];
cx q[4],q[5];
ry(1.5705747425381213) q[6];
ry(-2.900257387411288) q[7];
cx q[6],q[7];
ry(0.0043685242237021535) q[6];
ry(-1.6715666774086806) q[7];
cx q[6],q[7];
ry(0.8830970541418153) q[8];
ry(2.2321435153674654) q[9];
cx q[8],q[9];
ry(-2.9202626057304637) q[8];
ry(-1.1954640066344693) q[9];
cx q[8],q[9];
ry(2.645822807096484) q[10];
ry(-1.916123390198449) q[11];
cx q[10],q[11];
ry(1.7108320770575296) q[10];
ry(-0.6605949456958756) q[11];
cx q[10],q[11];
ry(0.7186863817408327) q[12];
ry(1.5749664980755476) q[13];
cx q[12],q[13];
ry(1.4333244943392445) q[12];
ry(3.059147487077123) q[13];
cx q[12],q[13];
ry(-0.3505497035958732) q[14];
ry(-3.037093528723229) q[15];
cx q[14],q[15];
ry(-1.3958229467668861) q[14];
ry(0.000764972501674066) q[15];
cx q[14],q[15];
ry(-0.03459206208941036) q[16];
ry(1.551323616538414) q[17];
cx q[16],q[17];
ry(-0.15623896165152248) q[16];
ry(3.1414144870480065) q[17];
cx q[16],q[17];
ry(-1.7442456696799837) q[18];
ry(2.433325010840711) q[19];
cx q[18],q[19];
ry(-3.1096151560063827) q[18];
ry(-2.39487314250774) q[19];
cx q[18],q[19];
ry(0.6339438781371225) q[1];
ry(1.6811140639034732) q[2];
cx q[1],q[2];
ry(3.004232562354877) q[1];
ry(-3.0511193400218337) q[2];
cx q[1],q[2];
ry(-2.8766757832467715) q[3];
ry(-1.2449601645013553) q[4];
cx q[3],q[4];
ry(-0.18714727778659856) q[3];
ry(-3.1415416793564734) q[4];
cx q[3],q[4];
ry(0.13940267942962556) q[5];
ry(2.4864307175051854) q[6];
cx q[5],q[6];
ry(1.1551008463350425) q[5];
ry(-0.5636572866696967) q[6];
cx q[5],q[6];
ry(-0.09145835711145534) q[7];
ry(-1.0028960693700464) q[8];
cx q[7],q[8];
ry(0.7176083034535357) q[7];
ry(2.2828493822337306) q[8];
cx q[7],q[8];
ry(2.120761228687075) q[9];
ry(0.6486132142345173) q[10];
cx q[9],q[10];
ry(-3.1278490526802916) q[9];
ry(3.1320569898102733) q[10];
cx q[9],q[10];
ry(0.5881384168200592) q[11];
ry(-2.9117975256729105) q[12];
cx q[11],q[12];
ry(-3.1351130801122693) q[11];
ry(-3.1396724775002376) q[12];
cx q[11],q[12];
ry(-1.5736114922236375) q[13];
ry(-2.3659141816181664) q[14];
cx q[13],q[14];
ry(2.880844668001113) q[13];
ry(1.4354759792616163) q[14];
cx q[13],q[14];
ry(-1.5721196016283445) q[15];
ry(2.6977862396863337) q[16];
cx q[15],q[16];
ry(2.801799163516764) q[15];
ry(-1.242808314435771) q[16];
cx q[15],q[16];
ry(-1.414686610679329) q[17];
ry(1.5186496906630627) q[18];
cx q[17],q[18];
ry(1.5077906613713132) q[17];
ry(2.717178542749375) q[18];
cx q[17],q[18];
ry(-2.92366333290033) q[0];
ry(-0.6068863663837921) q[1];
cx q[0],q[1];
ry(1.1579284973401909) q[0];
ry(0.8018962785804558) q[1];
cx q[0],q[1];
ry(2.5635652932259436) q[2];
ry(2.119334408345207) q[3];
cx q[2],q[3];
ry(-1.284722554951189) q[2];
ry(-2.8938918391837958) q[3];
cx q[2],q[3];
ry(0.98425946783741) q[4];
ry(-0.4679235047003253) q[5];
cx q[4],q[5];
ry(1.669415291204916) q[4];
ry(-0.7484686617367461) q[5];
cx q[4],q[5];
ry(0.25124311613394656) q[6];
ry(-0.4534976906340648) q[7];
cx q[6],q[7];
ry(4.1870589254340326e-05) q[6];
ry(3.1175106667310044) q[7];
cx q[6],q[7];
ry(1.639007103976159) q[8];
ry(-2.521498937092635) q[9];
cx q[8],q[9];
ry(2.7628546411232193) q[8];
ry(2.956686028536368) q[9];
cx q[8],q[9];
ry(1.866887001203338) q[10];
ry(-1.3233285013532186) q[11];
cx q[10],q[11];
ry(3.0071729142872927) q[10];
ry(0.060024620521358145) q[11];
cx q[10],q[11];
ry(-0.834584954306461) q[12];
ry(1.7317511793130669) q[13];
cx q[12],q[13];
ry(-3.119397054690944) q[12];
ry(-1.9342572332915617) q[13];
cx q[12],q[13];
ry(0.27621290587890995) q[14];
ry(-1.198040805497367) q[15];
cx q[14],q[15];
ry(0.24421956157625768) q[14];
ry(0.25036252717936147) q[15];
cx q[14],q[15];
ry(-2.11017121731017) q[16];
ry(1.9518592293848664) q[17];
cx q[16],q[17];
ry(-0.014614502880254676) q[16];
ry(-0.0022946123360130244) q[17];
cx q[16],q[17];
ry(-2.7330154139230025) q[18];
ry(-2.55037286643821) q[19];
cx q[18],q[19];
ry(-1.571011904321482) q[18];
ry(3.0018093465824585) q[19];
cx q[18],q[19];
ry(2.302145699862087) q[1];
ry(2.491579567328132) q[2];
cx q[1],q[2];
ry(1.6922572493154244) q[1];
ry(2.8773139056657566) q[2];
cx q[1],q[2];
ry(1.629601111326692) q[3];
ry(-1.568846280743788) q[4];
cx q[3],q[4];
ry(-0.0005194186717192294) q[3];
ry(-3.1414742469853105) q[4];
cx q[3],q[4];
ry(1.8138113758616408) q[5];
ry(-0.47427371254707396) q[6];
cx q[5],q[6];
ry(-2.7174751055457333) q[5];
ry(-1.3012429315010428) q[6];
cx q[5],q[6];
ry(3.04043375951523) q[7];
ry(0.22772438327261393) q[8];
cx q[7],q[8];
ry(-3.1380172147908034) q[7];
ry(-0.5953215191200996) q[8];
cx q[7],q[8];
ry(-1.4765691323324446) q[9];
ry(0.4078134648532031) q[10];
cx q[9],q[10];
ry(-0.0009717071604746679) q[9];
ry(0.0033574934273241652) q[10];
cx q[9],q[10];
ry(-2.1034307739788205) q[11];
ry(3.044424004248571) q[12];
cx q[11],q[12];
ry(3.118971409986652) q[11];
ry(1.764575810012576) q[12];
cx q[11],q[12];
ry(0.813214644932966) q[13];
ry(2.8384113811111655) q[14];
cx q[13],q[14];
ry(-2.7748468270150717) q[13];
ry(3.1291308407834317) q[14];
cx q[13],q[14];
ry(1.471466794301005) q[15];
ry(-1.0271170519096016) q[16];
cx q[15],q[16];
ry(-0.4239801409069083) q[15];
ry(-0.005276746089275322) q[16];
cx q[15],q[16];
ry(-0.052156827859920145) q[17];
ry(-0.160841517436622) q[18];
cx q[17],q[18];
ry(-0.08678135262467368) q[17];
ry(3.116423701261828) q[18];
cx q[17],q[18];
ry(-0.2961176848321214) q[0];
ry(-0.4596589776241703) q[1];
cx q[0],q[1];
ry(0.0053015425720950304) q[0];
ry(1.5151484152578962) q[1];
cx q[0],q[1];
ry(1.411849854611412) q[2];
ry(-1.5132241874271433) q[3];
cx q[2],q[3];
ry(1.3237212803005516) q[2];
ry(1.839474452537771) q[3];
cx q[2],q[3];
ry(-2.9040756184078407) q[4];
ry(1.761348891414765) q[5];
cx q[4],q[5];
ry(-2.3528526987983427) q[4];
ry(-1.2812041055241863) q[5];
cx q[4],q[5];
ry(-0.20136166869391292) q[6];
ry(-0.9527777193574272) q[7];
cx q[6],q[7];
ry(-3.1258536973992066) q[6];
ry(1.0200020047305811) q[7];
cx q[6],q[7];
ry(0.24474042429410578) q[8];
ry(0.8453935021539163) q[9];
cx q[8],q[9];
ry(2.622535129802198) q[8];
ry(2.4648617991214117) q[9];
cx q[8],q[9];
ry(-2.274824225463827) q[10];
ry(-1.5314978263884793) q[11];
cx q[10],q[11];
ry(1.118810489673935) q[10];
ry(-2.444024414092187) q[11];
cx q[10],q[11];
ry(0.5165246756033159) q[12];
ry(2.339928206662131) q[13];
cx q[12],q[13];
ry(-0.11148961907250499) q[12];
ry(-2.9726526661738) q[13];
cx q[12],q[13];
ry(1.4827915691384899) q[14];
ry(-1.354318623360486) q[15];
cx q[14],q[15];
ry(2.7722023864625527) q[14];
ry(-1.978446821046881) q[15];
cx q[14],q[15];
ry(-1.589534723264432) q[16];
ry(0.15616189804543623) q[17];
cx q[16],q[17];
ry(1.4396503700024752) q[16];
ry(1.4617551595963076) q[17];
cx q[16],q[17];
ry(-0.09939747549714628) q[18];
ry(-1.6356081634954984) q[19];
cx q[18],q[19];
ry(-0.30902251526878066) q[18];
ry(2.7023452901968876) q[19];
cx q[18],q[19];
ry(0.3761881301737624) q[1];
ry(-1.4843453869591963) q[2];
cx q[1],q[2];
ry(-1.9652460815001227) q[1];
ry(2.113007859189996) q[2];
cx q[1],q[2];
ry(1.569742683442994) q[3];
ry(-2.14988504045258) q[4];
cx q[3],q[4];
ry(3.1319477606210615) q[3];
ry(0.006081706966474787) q[4];
cx q[3],q[4];
ry(0.243096941027984) q[5];
ry(2.8481885303899594) q[6];
cx q[5],q[6];
ry(-3.140403120412655) q[5];
ry(-3.110516615719373) q[6];
cx q[5],q[6];
ry(-1.1805663139047748) q[7];
ry(1.6822247785477638) q[8];
cx q[7],q[8];
ry(1.1353449790601031) q[7];
ry(-3.1251316959939412) q[8];
cx q[7],q[8];
ry(-0.759252460471556) q[9];
ry(2.867138925452893) q[10];
cx q[9],q[10];
ry(-4.433663722728655e-05) q[9];
ry(0.42019283811095276) q[10];
cx q[9],q[10];
ry(2.353324754673951) q[11];
ry(-2.908267589125109) q[12];
cx q[11],q[12];
ry(3.1366852741104374) q[11];
ry(-0.0021785737286474744) q[12];
cx q[11],q[12];
ry(-0.5436775335273936) q[13];
ry(1.4992956777608768) q[14];
cx q[13],q[14];
ry(-1.3588497781006967) q[13];
ry(0.6897142605901606) q[14];
cx q[13],q[14];
ry(-1.4990291305670365) q[15];
ry(-2.334109644004619) q[16];
cx q[15],q[16];
ry(-0.6774044338858332) q[15];
ry(0.07926246590509223) q[16];
cx q[15],q[16];
ry(-2.6594726067846275) q[17];
ry(-1.7992011127233818) q[18];
cx q[17],q[18];
ry(-0.8333385101008739) q[17];
ry(3.128414581201895) q[18];
cx q[17],q[18];
ry(1.884413026637814) q[0];
ry(-1.4116791925156644) q[1];
cx q[0],q[1];
ry(1.457925048888752) q[0];
ry(1.4608316054673407) q[1];
cx q[0],q[1];
ry(-0.47967287291281036) q[2];
ry(0.2944582306301946) q[3];
cx q[2],q[3];
ry(-3.004865657022754) q[2];
ry(1.8633649481822485) q[3];
cx q[2],q[3];
ry(2.229848235748602) q[4];
ry(-2.0141404984188185) q[5];
cx q[4],q[5];
ry(-1.269058849203084) q[4];
ry(1.6620829274299096) q[5];
cx q[4],q[5];
ry(2.7432388448970744) q[6];
ry(-1.591678958155346) q[7];
cx q[6],q[7];
ry(-2.8771486699884945) q[6];
ry(1.0710410539870983) q[7];
cx q[6],q[7];
ry(-1.410459937236734) q[8];
ry(-1.5660458589168196) q[9];
cx q[8],q[9];
ry(-2.8150331141562415) q[8];
ry(0.030144166173544203) q[9];
cx q[8],q[9];
ry(-2.027686070310519) q[10];
ry(-2.6123604145876937) q[11];
cx q[10],q[11];
ry(-2.592748226175403) q[10];
ry(0.06775966230859164) q[11];
cx q[10],q[11];
ry(1.9449199860730326) q[12];
ry(0.7870618994390952) q[13];
cx q[12],q[13];
ry(-3.1149177012282183) q[12];
ry(-0.0012266262415217) q[13];
cx q[12],q[13];
ry(-1.5589450412684545) q[14];
ry(1.0104503349760376) q[15];
cx q[14],q[15];
ry(-0.005102053060956374) q[14];
ry(-0.2654284831440501) q[15];
cx q[14],q[15];
ry(-0.29682255684193654) q[16];
ry(2.4401664764695674) q[17];
cx q[16],q[17];
ry(3.1413700977771435) q[16];
ry(3.1410368986641695) q[17];
cx q[16],q[17];
ry(1.7631441444940636) q[18];
ry(-1.474881674435033) q[19];
cx q[18],q[19];
ry(0.09277647653943465) q[18];
ry(1.987900510435761) q[19];
cx q[18],q[19];
ry(2.12519826510245) q[1];
ry(1.8582997101287058) q[2];
cx q[1],q[2];
ry(-0.40746096030814716) q[1];
ry(-0.13903832222873636) q[2];
cx q[1],q[2];
ry(0.47017234353387255) q[3];
ry(1.1864402847439663) q[4];
cx q[3],q[4];
ry(-0.3854271920210728) q[3];
ry(-2.9915555277061854) q[4];
cx q[3],q[4];
ry(-1.3466828025574276) q[5];
ry(2.0653525642112487) q[6];
cx q[5],q[6];
ry(0.3516361483183603) q[5];
ry(0.08910628211421479) q[6];
cx q[5],q[6];
ry(-1.717199284189619) q[7];
ry(-1.805285027857387) q[8];
cx q[7],q[8];
ry(0.20828539093220044) q[7];
ry(-2.933072242827196) q[8];
cx q[7],q[8];
ry(-0.022864203297312997) q[9];
ry(2.3330186585658232) q[10];
cx q[9],q[10];
ry(2.9768469345941457) q[9];
ry(3.1183297982343987) q[10];
cx q[9],q[10];
ry(-2.483594033038931) q[11];
ry(-0.13600046862577653) q[12];
cx q[11],q[12];
ry(-2.8627092872220907) q[11];
ry(-1.899645738694316) q[12];
cx q[11],q[12];
ry(-2.362520044159262) q[13];
ry(1.5681218278383187) q[14];
cx q[13],q[14];
ry(2.0707020321824743) q[13];
ry(-0.2941306970990123) q[14];
cx q[13],q[14];
ry(2.2535933988161947) q[15];
ry(2.7838652228142657) q[16];
cx q[15],q[16];
ry(2.411386657292884) q[15];
ry(-0.3920082893646848) q[16];
cx q[15],q[16];
ry(0.803850715585237) q[17];
ry(1.9830590186221642) q[18];
cx q[17],q[18];
ry(1.2340755356694695) q[17];
ry(-2.7160648595807158) q[18];
cx q[17],q[18];
ry(-0.1504504946598031) q[0];
ry(2.594803017032858) q[1];
ry(-1.5637549812849438) q[2];
ry(0.07198115314231934) q[3];
ry(-1.5716363834565357) q[4];
ry(-0.022927848944164782) q[5];
ry(1.5530956171218901) q[6];
ry(3.0210860293267254) q[7];
ry(-1.5203431789923743) q[8];
ry(1.4869280968597056) q[9];
ry(1.5865204197989462) q[10];
ry(0.006303996059788554) q[11];
ry(-1.556764890010602) q[12];
ry(-3.130741870124295) q[13];
ry(-1.5719125548187352) q[14];
ry(0.013803744248119924) q[15];
ry(2.3062340845240565) q[16];
ry(-3.1368331098115156) q[17];
ry(-2.361502861461843) q[18];
ry(0.8205102731778932) q[19];