OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.6611672264675423) q[0];
ry(-2.377818105178395) q[1];
cx q[0],q[1];
ry(2.034034447554476) q[0];
ry(3.101230659694619) q[1];
cx q[0],q[1];
ry(0.7052492931576521) q[0];
ry(-1.7937197822715945) q[2];
cx q[0],q[2];
ry(-1.8175886268963872) q[0];
ry(1.6997341897537648) q[2];
cx q[0],q[2];
ry(1.0064431318001876) q[0];
ry(-2.1802695203537175) q[3];
cx q[0],q[3];
ry(1.6158488620924176) q[0];
ry(2.6620088338168477) q[3];
cx q[0],q[3];
ry(1.0213411302072029) q[0];
ry(-2.401397708384338) q[4];
cx q[0],q[4];
ry(2.733772853256815) q[0];
ry(-2.0255647156878) q[4];
cx q[0],q[4];
ry(2.5123112747851573) q[0];
ry(1.1183695882170839) q[5];
cx q[0],q[5];
ry(-1.4025912786044925) q[0];
ry(-0.6479572059992) q[5];
cx q[0],q[5];
ry(-1.9624630971035586) q[0];
ry(0.270631773532119) q[6];
cx q[0],q[6];
ry(2.848139758142163) q[0];
ry(-2.25553516268403) q[6];
cx q[0],q[6];
ry(0.22516822122482072) q[0];
ry(-0.5737324227635907) q[7];
cx q[0],q[7];
ry(2.2087227034819215) q[0];
ry(-2.856311938638026) q[7];
cx q[0],q[7];
ry(1.4510982076251726) q[0];
ry(0.4143642646337415) q[8];
cx q[0],q[8];
ry(-0.737976837849625) q[0];
ry(2.789994179246633) q[8];
cx q[0],q[8];
ry(-1.4469058973894822) q[0];
ry(-2.190459280502663) q[9];
cx q[0],q[9];
ry(-0.7308961611838384) q[0];
ry(-2.7660494000929448) q[9];
cx q[0],q[9];
ry(-0.7104064988122877) q[0];
ry(2.622211014083338) q[10];
cx q[0],q[10];
ry(0.04219003763345963) q[0];
ry(1.2570251603706852) q[10];
cx q[0],q[10];
ry(2.719056486148736) q[0];
ry(-0.7191944967581179) q[11];
cx q[0],q[11];
ry(2.3285255844462203) q[0];
ry(-0.05403766866154314) q[11];
cx q[0],q[11];
ry(-2.759087491734197) q[1];
ry(2.376274548551382) q[2];
cx q[1],q[2];
ry(1.5121846721738863) q[1];
ry(-1.6100465863262627) q[2];
cx q[1],q[2];
ry(1.032880555038827) q[1];
ry(-0.032030959593689) q[3];
cx q[1],q[3];
ry(1.1470893196317136) q[1];
ry(-1.3084623637610164) q[3];
cx q[1],q[3];
ry(0.1703345064275627) q[1];
ry(0.47623785127796836) q[4];
cx q[1],q[4];
ry(-2.0522268734832263) q[1];
ry(-2.0808344035603703) q[4];
cx q[1],q[4];
ry(0.7360618696362753) q[1];
ry(3.0657738128798213) q[5];
cx q[1],q[5];
ry(-2.1820155138103745) q[1];
ry(-1.4414815852482183) q[5];
cx q[1],q[5];
ry(-0.45464042754442335) q[1];
ry(2.591743841015224) q[6];
cx q[1],q[6];
ry(-2.124209101703437) q[1];
ry(-0.8218841888897858) q[6];
cx q[1],q[6];
ry(-1.3446661141056584) q[1];
ry(2.969924489772576) q[7];
cx q[1],q[7];
ry(-0.8167424158831998) q[1];
ry(-0.02876607621121341) q[7];
cx q[1],q[7];
ry(-1.1429588864118938) q[1];
ry(1.2343361831437925) q[8];
cx q[1],q[8];
ry(0.8954574470196547) q[1];
ry(-1.025081420329668) q[8];
cx q[1],q[8];
ry(-0.8398397851240963) q[1];
ry(2.791963962974224) q[9];
cx q[1],q[9];
ry(-0.5270001711151284) q[1];
ry(0.1203999479518707) q[9];
cx q[1],q[9];
ry(0.3143977659148624) q[1];
ry(0.6284524240634078) q[10];
cx q[1],q[10];
ry(0.9860024498381721) q[1];
ry(-1.9172016337759872) q[10];
cx q[1],q[10];
ry(-1.075128972438697) q[1];
ry(1.4885212858059684) q[11];
cx q[1],q[11];
ry(-1.5166839364388585) q[1];
ry(-1.0673347374749054) q[11];
cx q[1],q[11];
ry(2.9337845114358228) q[2];
ry(2.248871428740205) q[3];
cx q[2],q[3];
ry(2.993805120234863) q[2];
ry(-1.1944433525154414) q[3];
cx q[2],q[3];
ry(-1.5029906777630755) q[2];
ry(-2.6114645214522265) q[4];
cx q[2],q[4];
ry(-2.2827972183275813) q[2];
ry(1.3140223285584336) q[4];
cx q[2],q[4];
ry(-2.609615657716319) q[2];
ry(2.1124959867202593) q[5];
cx q[2],q[5];
ry(-0.5270253341719995) q[2];
ry(-0.9202433506621426) q[5];
cx q[2],q[5];
ry(0.45657119446537653) q[2];
ry(-2.7684447444865876) q[6];
cx q[2],q[6];
ry(-1.9560038380228506) q[2];
ry(2.776631605171109) q[6];
cx q[2],q[6];
ry(-2.328582801213131) q[2];
ry(1.575397666515076) q[7];
cx q[2],q[7];
ry(1.1734832238753645) q[2];
ry(-2.7527046508860034) q[7];
cx q[2],q[7];
ry(-1.312903657677725) q[2];
ry(-0.027614289241954852) q[8];
cx q[2],q[8];
ry(-0.8594329751437711) q[2];
ry(-1.7404660910267369) q[8];
cx q[2],q[8];
ry(1.8638006308460433) q[2];
ry(0.2944565465891574) q[9];
cx q[2],q[9];
ry(-2.0899918494006933) q[2];
ry(0.4391230110499853) q[9];
cx q[2],q[9];
ry(-1.6900166773710517) q[2];
ry(1.134733931389423) q[10];
cx q[2],q[10];
ry(2.5566454975926507) q[2];
ry(-0.4537166685350851) q[10];
cx q[2],q[10];
ry(2.8112527715605435) q[2];
ry(0.7150920430199901) q[11];
cx q[2],q[11];
ry(2.538887845197385) q[2];
ry(-0.30116383226949683) q[11];
cx q[2],q[11];
ry(-0.07989661530451433) q[3];
ry(-2.334185039668402) q[4];
cx q[3],q[4];
ry(-2.2912481818993222) q[3];
ry(1.78116861969683) q[4];
cx q[3],q[4];
ry(0.9903212098921408) q[3];
ry(2.257537782704845) q[5];
cx q[3],q[5];
ry(2.044380181061621) q[3];
ry(1.8652352814929734) q[5];
cx q[3],q[5];
ry(-0.8550719688329794) q[3];
ry(-2.993265844350745) q[6];
cx q[3],q[6];
ry(-1.8802367625845093) q[3];
ry(2.5886026324386577) q[6];
cx q[3],q[6];
ry(2.3822072667113976) q[3];
ry(-2.76608901285631) q[7];
cx q[3],q[7];
ry(-1.6396877193957307) q[3];
ry(1.2600291383692213) q[7];
cx q[3],q[7];
ry(2.924865910526695) q[3];
ry(0.6365950130485708) q[8];
cx q[3],q[8];
ry(-2.600719548565186) q[3];
ry(-0.10918555298681998) q[8];
cx q[3],q[8];
ry(0.034637143980887256) q[3];
ry(-1.323989754153648) q[9];
cx q[3],q[9];
ry(-2.9606570824903247) q[3];
ry(-0.5761481318816842) q[9];
cx q[3],q[9];
ry(2.8565109301651446) q[3];
ry(-0.34727204768414843) q[10];
cx q[3],q[10];
ry(-0.7906544167458307) q[3];
ry(1.3147422004480802) q[10];
cx q[3],q[10];
ry(-0.5723564097859963) q[3];
ry(-2.5563695515139573) q[11];
cx q[3],q[11];
ry(-1.7790211838594692) q[3];
ry(-0.19621711637998027) q[11];
cx q[3],q[11];
ry(-2.554252763711262) q[4];
ry(-1.2713350095190823) q[5];
cx q[4],q[5];
ry(0.25002033389029915) q[4];
ry(1.9260296648031627) q[5];
cx q[4],q[5];
ry(2.0024914017600994) q[4];
ry(-0.8403736037073929) q[6];
cx q[4],q[6];
ry(2.5757083091096438) q[4];
ry(-2.396382644480963) q[6];
cx q[4],q[6];
ry(2.209370151145976) q[4];
ry(-0.7968237332394199) q[7];
cx q[4],q[7];
ry(0.7959726562087394) q[4];
ry(2.471698577482485) q[7];
cx q[4],q[7];
ry(-0.13135623535069033) q[4];
ry(1.4731964168095448) q[8];
cx q[4],q[8];
ry(-0.29536790448791994) q[4];
ry(1.557557268998519) q[8];
cx q[4],q[8];
ry(-3.0009099288624888) q[4];
ry(2.7425839381012427) q[9];
cx q[4],q[9];
ry(2.8984236623853783) q[4];
ry(1.5830176839371983) q[9];
cx q[4],q[9];
ry(-1.742779228948896) q[4];
ry(2.135751802981482) q[10];
cx q[4],q[10];
ry(-3.0488166975196442) q[4];
ry(-0.22335887586518213) q[10];
cx q[4],q[10];
ry(2.5551177597195194) q[4];
ry(1.3859659066463454) q[11];
cx q[4],q[11];
ry(-0.32492258329552715) q[4];
ry(0.5642008429172402) q[11];
cx q[4],q[11];
ry(1.4530091687038935) q[5];
ry(-1.0007389893318903) q[6];
cx q[5],q[6];
ry(-2.603149976138943) q[5];
ry(-2.2994502907695518) q[6];
cx q[5],q[6];
ry(2.329710418459692) q[5];
ry(-3.0414359806301827) q[7];
cx q[5],q[7];
ry(0.17120371975145773) q[5];
ry(2.117895201817376) q[7];
cx q[5],q[7];
ry(1.700314098320133) q[5];
ry(0.6085345468528147) q[8];
cx q[5],q[8];
ry(-1.0677623865351293) q[5];
ry(-0.19380281599708113) q[8];
cx q[5],q[8];
ry(0.6557320457472784) q[5];
ry(-0.3657491610086868) q[9];
cx q[5],q[9];
ry(2.431620977432566) q[5];
ry(2.1077527114911394) q[9];
cx q[5],q[9];
ry(-2.1714701804141456) q[5];
ry(-1.8118258114403334) q[10];
cx q[5],q[10];
ry(-0.3975815077668994) q[5];
ry(-0.8325280327859081) q[10];
cx q[5],q[10];
ry(-0.3870862179331959) q[5];
ry(-0.3228260185096259) q[11];
cx q[5],q[11];
ry(-1.5203872724301295) q[5];
ry(2.841299710858395) q[11];
cx q[5],q[11];
ry(0.6675806550525181) q[6];
ry(-2.865747117326371) q[7];
cx q[6],q[7];
ry(2.258405620636448) q[6];
ry(-0.7151466052758936) q[7];
cx q[6],q[7];
ry(-1.0367104858339486) q[6];
ry(-1.7379913872393482) q[8];
cx q[6],q[8];
ry(-1.265091460795043) q[6];
ry(-2.8387818587602287) q[8];
cx q[6],q[8];
ry(2.7915880551167276) q[6];
ry(-0.4079377314197942) q[9];
cx q[6],q[9];
ry(1.4712191861573833) q[6];
ry(-2.1031448163903463) q[9];
cx q[6],q[9];
ry(-2.294250092491714) q[6];
ry(-1.457619485528281) q[10];
cx q[6],q[10];
ry(-0.30445560619348255) q[6];
ry(-0.10587599755115154) q[10];
cx q[6],q[10];
ry(0.10380498212736737) q[6];
ry(-3.0726744842068023) q[11];
cx q[6],q[11];
ry(-1.6787928237378653) q[6];
ry(-2.6433537408954084) q[11];
cx q[6],q[11];
ry(-1.4069560486906214) q[7];
ry(-2.638549385031796) q[8];
cx q[7],q[8];
ry(1.294876600599479) q[7];
ry(-2.528991568578963) q[8];
cx q[7],q[8];
ry(-0.9749184101331707) q[7];
ry(3.1254474279683517) q[9];
cx q[7],q[9];
ry(-0.558564917177258) q[7];
ry(-0.5263550734713887) q[9];
cx q[7],q[9];
ry(0.4531140069791144) q[7];
ry(0.8691139176879918) q[10];
cx q[7],q[10];
ry(0.8998011856322762) q[7];
ry(1.583623118439571) q[10];
cx q[7],q[10];
ry(2.4648269843830315) q[7];
ry(-2.033890884120982) q[11];
cx q[7],q[11];
ry(0.9386605327593405) q[7];
ry(0.4082646721554938) q[11];
cx q[7],q[11];
ry(1.3585498199490464) q[8];
ry(1.177350066446163) q[9];
cx q[8],q[9];
ry(0.6326514072134319) q[8];
ry(-2.33953074073941) q[9];
cx q[8],q[9];
ry(2.2897606936463686) q[8];
ry(-1.1611900073857377) q[10];
cx q[8],q[10];
ry(-2.3846172169767974) q[8];
ry(-2.28266460106586) q[10];
cx q[8],q[10];
ry(0.1688328531866187) q[8];
ry(1.6012768044131953) q[11];
cx q[8],q[11];
ry(1.0243233215109195) q[8];
ry(-1.6852823987959145) q[11];
cx q[8],q[11];
ry(-0.3696587157034106) q[9];
ry(-0.04896078090331876) q[10];
cx q[9],q[10];
ry(1.2594941607999819) q[9];
ry(1.9658431084845034) q[10];
cx q[9],q[10];
ry(-2.6716703356765605) q[9];
ry(0.8135065239026371) q[11];
cx q[9],q[11];
ry(-1.950255265984076) q[9];
ry(-0.174569096412343) q[11];
cx q[9],q[11];
ry(-0.603222570168839) q[10];
ry(1.6851671782027848) q[11];
cx q[10],q[11];
ry(-1.371463477938093) q[10];
ry(1.4825365468055862) q[11];
cx q[10],q[11];
ry(-1.7074881336334125) q[0];
ry(2.0858306223393255) q[1];
cx q[0],q[1];
ry(0.9058951942776812) q[0];
ry(-0.7862975781096587) q[1];
cx q[0],q[1];
ry(-1.2747736223232868) q[0];
ry(-1.7438880364231488) q[2];
cx q[0],q[2];
ry(1.0799054412705744) q[0];
ry(0.5192757356371693) q[2];
cx q[0],q[2];
ry(0.6871245407893839) q[0];
ry(2.6995562576966616) q[3];
cx q[0],q[3];
ry(2.0484254858181137) q[0];
ry(-0.8433970230972019) q[3];
cx q[0],q[3];
ry(1.4284654148023248) q[0];
ry(1.3125133783254952) q[4];
cx q[0],q[4];
ry(1.5458586437380821) q[0];
ry(-0.14114839150625297) q[4];
cx q[0],q[4];
ry(-2.8660531508322147) q[0];
ry(-0.40604668318173953) q[5];
cx q[0],q[5];
ry(-0.628119210086775) q[0];
ry(1.331505145152171) q[5];
cx q[0],q[5];
ry(-3.0116206216875843) q[0];
ry(-0.8061415998112738) q[6];
cx q[0],q[6];
ry(-1.6877409797967555) q[0];
ry(1.5991689999204004) q[6];
cx q[0],q[6];
ry(1.9864256152302635) q[0];
ry(0.7347837590648835) q[7];
cx q[0],q[7];
ry(2.7652123065858114) q[0];
ry(2.358100622102921) q[7];
cx q[0],q[7];
ry(-1.8121880881205739) q[0];
ry(2.9268281192590844) q[8];
cx q[0],q[8];
ry(-1.2838354164155978) q[0];
ry(0.054884034313190506) q[8];
cx q[0],q[8];
ry(-0.7657597924880104) q[0];
ry(0.5408453854508346) q[9];
cx q[0],q[9];
ry(-2.8530538089443276) q[0];
ry(2.157785146326235) q[9];
cx q[0],q[9];
ry(0.7238214265168041) q[0];
ry(-0.3593544869526306) q[10];
cx q[0],q[10];
ry(0.8792286953108976) q[0];
ry(2.355541276374137) q[10];
cx q[0],q[10];
ry(-0.7900373596093999) q[0];
ry(2.4029403208601634) q[11];
cx q[0],q[11];
ry(0.7507595173466851) q[0];
ry(2.584276236508755) q[11];
cx q[0],q[11];
ry(-0.7118769299585389) q[1];
ry(-1.019384756498103) q[2];
cx q[1],q[2];
ry(0.2749192504699841) q[1];
ry(2.7131665181676685) q[2];
cx q[1],q[2];
ry(0.18919178557291005) q[1];
ry(-2.651219963333811) q[3];
cx q[1],q[3];
ry(2.829844088838563) q[1];
ry(-0.6423539504517168) q[3];
cx q[1],q[3];
ry(0.09071911940570024) q[1];
ry(-1.2869998505021434) q[4];
cx q[1],q[4];
ry(-0.9301429322218411) q[1];
ry(1.2957327942689905) q[4];
cx q[1],q[4];
ry(2.903356566671418) q[1];
ry(0.28721986497255275) q[5];
cx q[1],q[5];
ry(2.710759293710742) q[1];
ry(-1.0396965093594506) q[5];
cx q[1],q[5];
ry(-0.49364825571787346) q[1];
ry(-0.3049681455815998) q[6];
cx q[1],q[6];
ry(-2.5493848185699495) q[1];
ry(-1.4957219539489255) q[6];
cx q[1],q[6];
ry(-0.29455581164550854) q[1];
ry(-1.9907881514811065) q[7];
cx q[1],q[7];
ry(-1.1255687657217721) q[1];
ry(0.4045744593139503) q[7];
cx q[1],q[7];
ry(1.9368474032912628) q[1];
ry(-2.1466825845830293) q[8];
cx q[1],q[8];
ry(1.3211444401184105) q[1];
ry(-1.6221890727456536) q[8];
cx q[1],q[8];
ry(-2.112933300046258) q[1];
ry(-1.2579983560002335) q[9];
cx q[1],q[9];
ry(-0.907463462122621) q[1];
ry(2.333362611504673) q[9];
cx q[1],q[9];
ry(0.5779763640938936) q[1];
ry(-0.21796917967404728) q[10];
cx q[1],q[10];
ry(0.7926340105222113) q[1];
ry(0.5086698021755032) q[10];
cx q[1],q[10];
ry(-2.0904047358996953) q[1];
ry(-2.088141875704954) q[11];
cx q[1],q[11];
ry(2.5015864110478345) q[1];
ry(-0.9981296067459099) q[11];
cx q[1],q[11];
ry(-2.313108971823619) q[2];
ry(0.9833344542591426) q[3];
cx q[2],q[3];
ry(-2.2604760159624444) q[2];
ry(2.2551703062998962) q[3];
cx q[2],q[3];
ry(2.8227004867798415) q[2];
ry(-0.7787401080361569) q[4];
cx q[2],q[4];
ry(2.823520533709054) q[2];
ry(-2.0663281256759034) q[4];
cx q[2],q[4];
ry(-3.0146866143823994) q[2];
ry(2.4338721313227523) q[5];
cx q[2],q[5];
ry(2.5544331371869275) q[2];
ry(0.6017260173383692) q[5];
cx q[2],q[5];
ry(0.47107400167516467) q[2];
ry(1.8589770469217504) q[6];
cx q[2],q[6];
ry(-1.034941738986224) q[2];
ry(0.7288882681269029) q[6];
cx q[2],q[6];
ry(2.2321764133984425) q[2];
ry(-2.7489233234924235) q[7];
cx q[2],q[7];
ry(-2.985290029938399) q[2];
ry(-2.6854169581931444) q[7];
cx q[2],q[7];
ry(-0.5997845376082643) q[2];
ry(-2.302132592087795) q[8];
cx q[2],q[8];
ry(-0.9320035386473801) q[2];
ry(2.600094891117125) q[8];
cx q[2],q[8];
ry(-1.3036143743160162) q[2];
ry(3.02586280271045) q[9];
cx q[2],q[9];
ry(-0.018773272662002682) q[2];
ry(2.384546643628103) q[9];
cx q[2],q[9];
ry(1.1221861408486733) q[2];
ry(-2.956637261410395) q[10];
cx q[2],q[10];
ry(2.022100562301985) q[2];
ry(-1.371010050087666) q[10];
cx q[2],q[10];
ry(1.5682534893600373) q[2];
ry(-1.8573499144373742) q[11];
cx q[2],q[11];
ry(-1.68426463909009) q[2];
ry(1.675517203998248) q[11];
cx q[2],q[11];
ry(-1.2498192499729308) q[3];
ry(-0.8499502939766154) q[4];
cx q[3],q[4];
ry(1.8618118973054836) q[3];
ry(-2.5380833595419268) q[4];
cx q[3],q[4];
ry(-2.748647772068634) q[3];
ry(-1.701936326495274) q[5];
cx q[3],q[5];
ry(-3.0680765863934365) q[3];
ry(1.1122712255325993) q[5];
cx q[3],q[5];
ry(0.2174132231442023) q[3];
ry(-2.410101889457149) q[6];
cx q[3],q[6];
ry(1.5368759115930963) q[3];
ry(-3.101004783051143) q[6];
cx q[3],q[6];
ry(-1.526049101460839) q[3];
ry(2.8037501341454045) q[7];
cx q[3],q[7];
ry(-2.205785182734493) q[3];
ry(1.466589134490026) q[7];
cx q[3],q[7];
ry(-3.0353617155682455) q[3];
ry(-2.589401292911205) q[8];
cx q[3],q[8];
ry(-2.8432776868451053) q[3];
ry(2.5773209853976424) q[8];
cx q[3],q[8];
ry(-0.074892705875099) q[3];
ry(1.6691654510616623) q[9];
cx q[3],q[9];
ry(-2.7864347080239837) q[3];
ry(-2.989074254806766) q[9];
cx q[3],q[9];
ry(-1.7492869761007646) q[3];
ry(1.2265009644990954) q[10];
cx q[3],q[10];
ry(0.7771318382462767) q[3];
ry(-1.8182931036230683) q[10];
cx q[3],q[10];
ry(1.5731298542013858) q[3];
ry(2.2011834315523284) q[11];
cx q[3],q[11];
ry(1.5102032039722681) q[3];
ry(-1.0123112822957745) q[11];
cx q[3],q[11];
ry(0.6374876553785505) q[4];
ry(1.4533531526948262) q[5];
cx q[4],q[5];
ry(0.9164709952916054) q[4];
ry(-0.3353031136176934) q[5];
cx q[4],q[5];
ry(0.31566786830909294) q[4];
ry(0.547189776148698) q[6];
cx q[4],q[6];
ry(3.066628894849646) q[4];
ry(-0.9085791521587634) q[6];
cx q[4],q[6];
ry(0.14094757039374084) q[4];
ry(0.39447575260588036) q[7];
cx q[4],q[7];
ry(-1.622931069244566) q[4];
ry(-0.15225565248588555) q[7];
cx q[4],q[7];
ry(-2.1561670199132856) q[4];
ry(-2.8489280574297875) q[8];
cx q[4],q[8];
ry(-2.5801938936701) q[4];
ry(-2.6909630000205955) q[8];
cx q[4],q[8];
ry(2.2657170922264296) q[4];
ry(-0.2026572838118641) q[9];
cx q[4],q[9];
ry(0.9860325617059337) q[4];
ry(0.43498800122299386) q[9];
cx q[4],q[9];
ry(-0.6830851056600977) q[4];
ry(-2.7104579600261816) q[10];
cx q[4],q[10];
ry(-1.6889720715062628) q[4];
ry(1.4533358248916208) q[10];
cx q[4],q[10];
ry(0.3315034101170167) q[4];
ry(-0.6101887787072823) q[11];
cx q[4],q[11];
ry(2.1048683592195037) q[4];
ry(-0.9328578250385416) q[11];
cx q[4],q[11];
ry(-0.7667660068235393) q[5];
ry(2.7642085783477977) q[6];
cx q[5],q[6];
ry(1.2284358749143816) q[5];
ry(1.469778765375566) q[6];
cx q[5],q[6];
ry(2.6925129014916966) q[5];
ry(0.4804417382503985) q[7];
cx q[5],q[7];
ry(0.10768465580408904) q[5];
ry(-2.53766224432093) q[7];
cx q[5],q[7];
ry(1.1252942312926573) q[5];
ry(2.894755032714748) q[8];
cx q[5],q[8];
ry(-1.7645448825995045) q[5];
ry(-0.7113156053074653) q[8];
cx q[5],q[8];
ry(1.0008779816405904) q[5];
ry(0.3990004101026461) q[9];
cx q[5],q[9];
ry(0.8786400708156805) q[5];
ry(-2.1109814839785446) q[9];
cx q[5],q[9];
ry(3.0283684558233235) q[5];
ry(-0.8085509963607471) q[10];
cx q[5],q[10];
ry(-0.5163350084747478) q[5];
ry(0.2582469733713255) q[10];
cx q[5],q[10];
ry(-0.018847713843964442) q[5];
ry(2.6565690724057585) q[11];
cx q[5],q[11];
ry(2.941120404003301) q[5];
ry(0.8738524732609791) q[11];
cx q[5],q[11];
ry(-1.0433100505768893) q[6];
ry(1.4354505952479597) q[7];
cx q[6],q[7];
ry(-0.9233685325613823) q[6];
ry(0.06742384843747078) q[7];
cx q[6],q[7];
ry(0.38184158577245775) q[6];
ry(-2.0149536781564965) q[8];
cx q[6],q[8];
ry(0.4719785706064797) q[6];
ry(2.048964101521903) q[8];
cx q[6],q[8];
ry(1.1599865428301637) q[6];
ry(0.16266929677766942) q[9];
cx q[6],q[9];
ry(1.9001083405752657) q[6];
ry(0.7010818229248016) q[9];
cx q[6],q[9];
ry(0.22749222337422736) q[6];
ry(0.6418223387266888) q[10];
cx q[6],q[10];
ry(1.7354449541942978) q[6];
ry(-0.9339013201310836) q[10];
cx q[6],q[10];
ry(1.5667360720478476) q[6];
ry(0.6211366700177718) q[11];
cx q[6],q[11];
ry(-1.9190276011583305) q[6];
ry(-1.4638383161589288) q[11];
cx q[6],q[11];
ry(-0.24012606277647566) q[7];
ry(-1.881483107003958) q[8];
cx q[7],q[8];
ry(-1.5300783909214186) q[7];
ry(0.6953058478525281) q[8];
cx q[7],q[8];
ry(-1.0990874071970902) q[7];
ry(-2.4421995953438027) q[9];
cx q[7],q[9];
ry(1.1655114320834412) q[7];
ry(1.7864905104259377) q[9];
cx q[7],q[9];
ry(2.128432949009172) q[7];
ry(-1.9693430115029553) q[10];
cx q[7],q[10];
ry(-1.2510779258139142) q[7];
ry(0.8917719876941801) q[10];
cx q[7],q[10];
ry(-2.812951936254692) q[7];
ry(0.016317171843432784) q[11];
cx q[7],q[11];
ry(-0.3163169509969348) q[7];
ry(2.1509513212211466) q[11];
cx q[7],q[11];
ry(2.0020964261630687) q[8];
ry(-1.9780762427359493) q[9];
cx q[8],q[9];
ry(0.9598926632209901) q[8];
ry(2.2129835590692366) q[9];
cx q[8],q[9];
ry(-2.4467478195760757) q[8];
ry(1.2513306083183409) q[10];
cx q[8],q[10];
ry(2.282519832622154) q[8];
ry(-1.9173970494184422) q[10];
cx q[8],q[10];
ry(3.058351882909458) q[8];
ry(-1.0962618475580932) q[11];
cx q[8],q[11];
ry(-1.548305480389794) q[8];
ry(2.5669098495349383) q[11];
cx q[8],q[11];
ry(0.9974387069025283) q[9];
ry(-2.8469816009300795) q[10];
cx q[9],q[10];
ry(-0.39079812777984463) q[9];
ry(-1.6144758506182806) q[10];
cx q[9],q[10];
ry(0.9649709355097597) q[9];
ry(-0.11684982733469072) q[11];
cx q[9],q[11];
ry(-3.034889617492089) q[9];
ry(1.5135603144465424) q[11];
cx q[9],q[11];
ry(-2.074232233961198) q[10];
ry(0.7704427690154478) q[11];
cx q[10],q[11];
ry(-2.914269733176023) q[10];
ry(2.6679301768609576) q[11];
cx q[10],q[11];
ry(1.149616912668297) q[0];
ry(0.6028478805833108) q[1];
cx q[0],q[1];
ry(-1.7129051940841515) q[0];
ry(-2.5882372131194464) q[1];
cx q[0],q[1];
ry(-0.47429933900250537) q[0];
ry(-2.4463044311393447) q[2];
cx q[0],q[2];
ry(2.8484091344052893) q[0];
ry(1.2450855710117663) q[2];
cx q[0],q[2];
ry(1.123868495643892) q[0];
ry(1.1010766630522921) q[3];
cx q[0],q[3];
ry(-0.5323782335365419) q[0];
ry(2.206481552121013) q[3];
cx q[0],q[3];
ry(1.6018626374445808) q[0];
ry(1.9413751269465243) q[4];
cx q[0],q[4];
ry(1.6882788314266683) q[0];
ry(-0.6304266792842912) q[4];
cx q[0],q[4];
ry(2.266327708230194) q[0];
ry(1.1743612328664315) q[5];
cx q[0],q[5];
ry(1.7661016645811167) q[0];
ry(2.645339779973159) q[5];
cx q[0],q[5];
ry(3.1029570665356525) q[0];
ry(-1.2543158520540425) q[6];
cx q[0],q[6];
ry(-2.442807785721248) q[0];
ry(-2.319878233697676) q[6];
cx q[0],q[6];
ry(0.6979993710750723) q[0];
ry(-0.02854097161362503) q[7];
cx q[0],q[7];
ry(1.8310623259745438) q[0];
ry(-1.4075388922692267) q[7];
cx q[0],q[7];
ry(-2.3593313640240674) q[0];
ry(2.0944104571006426) q[8];
cx q[0],q[8];
ry(1.2485375531517566) q[0];
ry(0.3288270994871141) q[8];
cx q[0],q[8];
ry(-2.5603114618852967) q[0];
ry(-0.503721444989842) q[9];
cx q[0],q[9];
ry(-2.296420754954605) q[0];
ry(-2.543360637148825) q[9];
cx q[0],q[9];
ry(-2.7258098564615794) q[0];
ry(0.08162512004134202) q[10];
cx q[0],q[10];
ry(-1.2694014956165525) q[0];
ry(0.6219937619078015) q[10];
cx q[0],q[10];
ry(-0.8807009840793905) q[0];
ry(-1.9228368663788116) q[11];
cx q[0],q[11];
ry(0.76293283329764) q[0];
ry(2.6547064827033977) q[11];
cx q[0],q[11];
ry(-1.0834299937352962) q[1];
ry(-0.0988706189402864) q[2];
cx q[1],q[2];
ry(0.10894066100652945) q[1];
ry(-2.066671630777634) q[2];
cx q[1],q[2];
ry(1.614798629476806) q[1];
ry(0.8175753289906239) q[3];
cx q[1],q[3];
ry(-1.178229759980976) q[1];
ry(-1.7281984117458524) q[3];
cx q[1],q[3];
ry(-2.853105591561273) q[1];
ry(-2.2062711157438626) q[4];
cx q[1],q[4];
ry(1.9271613772352234) q[1];
ry(-2.1630006658290406) q[4];
cx q[1],q[4];
ry(-2.4426083516138783) q[1];
ry(-2.122581523293622) q[5];
cx q[1],q[5];
ry(2.6516806161224404) q[1];
ry(-0.9429349314022892) q[5];
cx q[1],q[5];
ry(1.1244889248787227) q[1];
ry(-1.1577060853488512) q[6];
cx q[1],q[6];
ry(-0.3035188075431725) q[1];
ry(1.3679368486947805) q[6];
cx q[1],q[6];
ry(1.0673709030855854) q[1];
ry(1.1973420206435583) q[7];
cx q[1],q[7];
ry(2.8465512106711106) q[1];
ry(-1.1093165887454701) q[7];
cx q[1],q[7];
ry(-0.2840990930247655) q[1];
ry(-2.9868107196836275) q[8];
cx q[1],q[8];
ry(0.9996215794908441) q[1];
ry(1.964014591579153) q[8];
cx q[1],q[8];
ry(-2.750750904581138) q[1];
ry(0.031102271857084144) q[9];
cx q[1],q[9];
ry(1.8659376945417963) q[1];
ry(2.277472898910485) q[9];
cx q[1],q[9];
ry(-1.656413139797645) q[1];
ry(-1.0584492336418485) q[10];
cx q[1],q[10];
ry(-2.5005451864311965) q[1];
ry(2.3547044750760793) q[10];
cx q[1],q[10];
ry(2.6401542729991796) q[1];
ry(0.9122543056539143) q[11];
cx q[1],q[11];
ry(-1.469071237391823) q[1];
ry(1.098576967991752) q[11];
cx q[1],q[11];
ry(1.5506781501676008) q[2];
ry(1.377429994900159) q[3];
cx q[2],q[3];
ry(-1.0710622721195966) q[2];
ry(-1.5466200049051524) q[3];
cx q[2],q[3];
ry(-1.9755701513554509) q[2];
ry(0.3632738074011712) q[4];
cx q[2],q[4];
ry(2.181332092954694) q[2];
ry(2.7232930753917337) q[4];
cx q[2],q[4];
ry(-0.981536370183337) q[2];
ry(-0.4075008146151954) q[5];
cx q[2],q[5];
ry(1.4905946055071215) q[2];
ry(0.6556837594121354) q[5];
cx q[2],q[5];
ry(-2.639917298472774) q[2];
ry(-2.2105152704887168) q[6];
cx q[2],q[6];
ry(-0.46322365823990896) q[2];
ry(-1.963012133648781) q[6];
cx q[2],q[6];
ry(0.13664913905537546) q[2];
ry(-1.7889167055703572) q[7];
cx q[2],q[7];
ry(-1.074576843822865) q[2];
ry(2.9077800731859687) q[7];
cx q[2],q[7];
ry(-0.6939272398644996) q[2];
ry(-2.5387484787466392) q[8];
cx q[2],q[8];
ry(-1.8016613955182403) q[2];
ry(2.3707257560886226) q[8];
cx q[2],q[8];
ry(1.386766582365624) q[2];
ry(-0.8829422064312551) q[9];
cx q[2],q[9];
ry(-1.2388406389948161) q[2];
ry(0.7260598408766245) q[9];
cx q[2],q[9];
ry(2.9969056068087427) q[2];
ry(-1.337659504344333) q[10];
cx q[2],q[10];
ry(-1.6660288391269564) q[2];
ry(2.164373431923358) q[10];
cx q[2],q[10];
ry(1.0352535080513503) q[2];
ry(1.7323795839033718) q[11];
cx q[2],q[11];
ry(3.001209078588814) q[2];
ry(-0.5094302290522271) q[11];
cx q[2],q[11];
ry(-2.139026096073801) q[3];
ry(-2.8474678533261377) q[4];
cx q[3],q[4];
ry(-0.13925293098066305) q[3];
ry(-1.5673520322160694) q[4];
cx q[3],q[4];
ry(-2.9639419916866747) q[3];
ry(-2.426440414821803) q[5];
cx q[3],q[5];
ry(0.5358973116397068) q[3];
ry(0.7189761646505917) q[5];
cx q[3],q[5];
ry(0.8313739373244192) q[3];
ry(2.4751665170085633) q[6];
cx q[3],q[6];
ry(1.8793293094675074) q[3];
ry(1.719845028544475) q[6];
cx q[3],q[6];
ry(-1.9949439158876396) q[3];
ry(0.3862668015179977) q[7];
cx q[3],q[7];
ry(-1.8194470715001385) q[3];
ry(-2.261274868303412) q[7];
cx q[3],q[7];
ry(0.9324001116721492) q[3];
ry(-1.8607496822857073) q[8];
cx q[3],q[8];
ry(-1.228122850142323) q[3];
ry(-1.4059354340206953) q[8];
cx q[3],q[8];
ry(0.9087138106280973) q[3];
ry(1.295516374773503) q[9];
cx q[3],q[9];
ry(-0.4386289827696217) q[3];
ry(1.7454555725600764) q[9];
cx q[3],q[9];
ry(-1.8641825533005056) q[3];
ry(-1.4493413735404497) q[10];
cx q[3],q[10];
ry(-1.9431812534901223) q[3];
ry(-1.3001787505289721) q[10];
cx q[3],q[10];
ry(1.2057215545611237) q[3];
ry(-1.1992141958374698) q[11];
cx q[3],q[11];
ry(1.1415917554243402) q[3];
ry(-1.9151454005987618) q[11];
cx q[3],q[11];
ry(2.7573523610491626) q[4];
ry(-2.618637658577531) q[5];
cx q[4],q[5];
ry(-2.2531229204138903) q[4];
ry(-0.7107005026003732) q[5];
cx q[4],q[5];
ry(-0.1431119766116416) q[4];
ry(-1.525391895501671) q[6];
cx q[4],q[6];
ry(-2.1699546099955915) q[4];
ry(0.7463563164522933) q[6];
cx q[4],q[6];
ry(1.5658487758272592) q[4];
ry(-0.8193945163067928) q[7];
cx q[4],q[7];
ry(2.410064243247714) q[4];
ry(-1.520284385760973) q[7];
cx q[4],q[7];
ry(0.41267810348757467) q[4];
ry(-0.11654420229790574) q[8];
cx q[4],q[8];
ry(-0.4991653346567526) q[4];
ry(-0.55551343646792) q[8];
cx q[4],q[8];
ry(0.9089653134025307) q[4];
ry(0.16206861324654567) q[9];
cx q[4],q[9];
ry(2.852673925652234) q[4];
ry(-1.763748369460873) q[9];
cx q[4],q[9];
ry(-0.3649754938465481) q[4];
ry(-2.5657828944226413) q[10];
cx q[4],q[10];
ry(2.0007194849762127) q[4];
ry(-2.494125662822498) q[10];
cx q[4],q[10];
ry(0.34383604171546) q[4];
ry(2.6328452697148252) q[11];
cx q[4],q[11];
ry(1.5068568176776003) q[4];
ry(2.534067207073406) q[11];
cx q[4],q[11];
ry(-0.3191624290094213) q[5];
ry(1.465706139747871) q[6];
cx q[5],q[6];
ry(-2.258088624815206) q[5];
ry(2.3048036781168135) q[6];
cx q[5],q[6];
ry(-2.2988789217045715) q[5];
ry(2.445551214307582) q[7];
cx q[5],q[7];
ry(2.54367083396722) q[5];
ry(-1.4691357620126733) q[7];
cx q[5],q[7];
ry(-2.0926725294691617) q[5];
ry(-2.39245849565386) q[8];
cx q[5],q[8];
ry(-3.011348319467012) q[5];
ry(1.831379836321508) q[8];
cx q[5],q[8];
ry(1.7435543663288433) q[5];
ry(0.6318382648667592) q[9];
cx q[5],q[9];
ry(-1.9241930492278225) q[5];
ry(-1.1656844600879896) q[9];
cx q[5],q[9];
ry(-1.909696281875621) q[5];
ry(-0.6014839181464766) q[10];
cx q[5],q[10];
ry(-2.9286806315762717) q[5];
ry(-0.9412957700911402) q[10];
cx q[5],q[10];
ry(2.100463029041478) q[5];
ry(2.3118364086044587) q[11];
cx q[5],q[11];
ry(-0.07803572328017426) q[5];
ry(1.933027600682708) q[11];
cx q[5],q[11];
ry(-0.3411688469328418) q[6];
ry(1.7279846375681105) q[7];
cx q[6],q[7];
ry(-0.4702836400874899) q[6];
ry(-1.5854606575775754) q[7];
cx q[6],q[7];
ry(-1.4693276271183477) q[6];
ry(-3.074661365334833) q[8];
cx q[6],q[8];
ry(1.9865359540580325) q[6];
ry(0.28041553293421106) q[8];
cx q[6],q[8];
ry(-2.722165364510401) q[6];
ry(1.9267555082141572) q[9];
cx q[6],q[9];
ry(-2.0694648494457244) q[6];
ry(3.0805171826714397) q[9];
cx q[6],q[9];
ry(0.7543607589696402) q[6];
ry(0.920308796359614) q[10];
cx q[6],q[10];
ry(-0.3380319516262826) q[6];
ry(0.6111410578517846) q[10];
cx q[6],q[10];
ry(-1.246957701949304) q[6];
ry(-2.588738848779648) q[11];
cx q[6],q[11];
ry(0.7310405328262577) q[6];
ry(0.23856330603491488) q[11];
cx q[6],q[11];
ry(0.3304778186616719) q[7];
ry(-2.6276675733340036) q[8];
cx q[7],q[8];
ry(2.085876986162768) q[7];
ry(-0.317805748351371) q[8];
cx q[7],q[8];
ry(1.4270995021263007) q[7];
ry(-0.40466102345102517) q[9];
cx q[7],q[9];
ry(1.893098177579634) q[7];
ry(1.8254330355444992) q[9];
cx q[7],q[9];
ry(-0.05758632160504896) q[7];
ry(1.0665557801080974) q[10];
cx q[7],q[10];
ry(2.064302431202565) q[7];
ry(0.5394251227954026) q[10];
cx q[7],q[10];
ry(-1.1261198558870005) q[7];
ry(0.46415203631127233) q[11];
cx q[7],q[11];
ry(-0.20628175881487448) q[7];
ry(0.8794544746086222) q[11];
cx q[7],q[11];
ry(0.22267140597840004) q[8];
ry(2.484060684601927) q[9];
cx q[8],q[9];
ry(-2.6756456482482354) q[8];
ry(-1.200291237904627) q[9];
cx q[8],q[9];
ry(-2.569487009639316) q[8];
ry(-3.059761841175372) q[10];
cx q[8],q[10];
ry(2.3132596282735167) q[8];
ry(-3.0290177529190108) q[10];
cx q[8],q[10];
ry(-1.6444145627239477) q[8];
ry(-1.7068083556095326) q[11];
cx q[8],q[11];
ry(0.9622360306649282) q[8];
ry(-0.08679461126440824) q[11];
cx q[8],q[11];
ry(0.9063866408926283) q[9];
ry(-2.420779863726897) q[10];
cx q[9],q[10];
ry(2.4554969775039144) q[9];
ry(-0.37265989190677296) q[10];
cx q[9],q[10];
ry(1.0720199759316837) q[9];
ry(1.456315352002136) q[11];
cx q[9],q[11];
ry(-1.1552507362135023) q[9];
ry(1.753389642521154) q[11];
cx q[9],q[11];
ry(-0.21352114374789963) q[10];
ry(0.7383967071859309) q[11];
cx q[10],q[11];
ry(-1.207171841254121) q[10];
ry(-0.8725189119379904) q[11];
cx q[10],q[11];
ry(0.20071735460259663) q[0];
ry(1.3033125080601664) q[1];
cx q[0],q[1];
ry(1.3800016629151726) q[0];
ry(-2.8431470045148757) q[1];
cx q[0],q[1];
ry(1.5442366133622816) q[0];
ry(1.046466954408897) q[2];
cx q[0],q[2];
ry(2.354723627292897) q[0];
ry(-1.4872310805037183) q[2];
cx q[0],q[2];
ry(-2.0037840752299534) q[0];
ry(-1.0812208629801272) q[3];
cx q[0],q[3];
ry(2.952864072537017) q[0];
ry(0.055995297571529934) q[3];
cx q[0],q[3];
ry(0.29316911177014776) q[0];
ry(-1.2316710411616814) q[4];
cx q[0],q[4];
ry(-2.792007248622575) q[0];
ry(-2.2450506617075305) q[4];
cx q[0],q[4];
ry(1.4208932093575177) q[0];
ry(-0.571788065200316) q[5];
cx q[0],q[5];
ry(-0.9034230456745247) q[0];
ry(-0.3815003415469782) q[5];
cx q[0],q[5];
ry(-2.9156301708501196) q[0];
ry(-2.8957264776022784) q[6];
cx q[0],q[6];
ry(1.4422129601157039) q[0];
ry(-1.7541985324306246) q[6];
cx q[0],q[6];
ry(-1.0988117431000868) q[0];
ry(-1.2041390141724584) q[7];
cx q[0],q[7];
ry(-2.5424223549133207) q[0];
ry(-2.4183265258464584) q[7];
cx q[0],q[7];
ry(-1.8365132720325779) q[0];
ry(-1.3142636688765492) q[8];
cx q[0],q[8];
ry(-1.088016012941864) q[0];
ry(-1.3168217657922758) q[8];
cx q[0],q[8];
ry(-0.2703456489858663) q[0];
ry(2.906456809007722) q[9];
cx q[0],q[9];
ry(0.1065917301143342) q[0];
ry(-2.534144172928773) q[9];
cx q[0],q[9];
ry(-1.4684142288814463) q[0];
ry(-1.4946607518576203) q[10];
cx q[0],q[10];
ry(-0.32261527072642476) q[0];
ry(2.260609946697981) q[10];
cx q[0],q[10];
ry(-2.396767830535553) q[0];
ry(2.996348561817541) q[11];
cx q[0],q[11];
ry(-1.3827493103378388) q[0];
ry(-2.6323468601135125) q[11];
cx q[0],q[11];
ry(2.1555155162141393) q[1];
ry(1.4693055942512823) q[2];
cx q[1],q[2];
ry(-2.6792313505687018) q[1];
ry(1.1334694273629446) q[2];
cx q[1],q[2];
ry(0.9074643030341406) q[1];
ry(0.045596144438320045) q[3];
cx q[1],q[3];
ry(-1.851841852764781) q[1];
ry(-0.6119390600798491) q[3];
cx q[1],q[3];
ry(1.0090676782259322) q[1];
ry(-2.5834491087847646) q[4];
cx q[1],q[4];
ry(1.383937148283727) q[1];
ry(-1.224115163074645) q[4];
cx q[1],q[4];
ry(3.0132163789002546) q[1];
ry(1.3357127508586513) q[5];
cx q[1],q[5];
ry(1.7781292157785364) q[1];
ry(-0.9764170852511304) q[5];
cx q[1],q[5];
ry(0.6504592031787455) q[1];
ry(3.0202317800205636) q[6];
cx q[1],q[6];
ry(-2.609425102873362) q[1];
ry(-0.4841501522090681) q[6];
cx q[1],q[6];
ry(-1.944507096145895) q[1];
ry(-2.6345622607003367) q[7];
cx q[1],q[7];
ry(1.005006369014521) q[1];
ry(-0.6713977572791763) q[7];
cx q[1],q[7];
ry(1.288730135925702) q[1];
ry(-1.770235533522463) q[8];
cx q[1],q[8];
ry(0.3664859234310911) q[1];
ry(-1.0642501165029092) q[8];
cx q[1],q[8];
ry(2.5211644903131756) q[1];
ry(0.2554267473846148) q[9];
cx q[1],q[9];
ry(-2.6799911978417557) q[1];
ry(1.3155892707076706) q[9];
cx q[1],q[9];
ry(1.0614846486009375) q[1];
ry(-0.1639412984952022) q[10];
cx q[1],q[10];
ry(1.6932774494457108) q[1];
ry(1.526188271623575) q[10];
cx q[1],q[10];
ry(1.549919049598972) q[1];
ry(-1.334424999246226) q[11];
cx q[1],q[11];
ry(2.555979350358591) q[1];
ry(-1.411999385603067) q[11];
cx q[1],q[11];
ry(-2.886285147345811) q[2];
ry(-0.812336181235284) q[3];
cx q[2],q[3];
ry(-1.7767144068216156) q[2];
ry(-2.4465689369606536) q[3];
cx q[2],q[3];
ry(2.049727876941999) q[2];
ry(2.468808551355106) q[4];
cx q[2],q[4];
ry(-0.2319195176638411) q[2];
ry(0.39777453743670943) q[4];
cx q[2],q[4];
ry(0.7419552667680498) q[2];
ry(1.8440045287750548) q[5];
cx q[2],q[5];
ry(-2.5797397478862756) q[2];
ry(1.1066246601151564) q[5];
cx q[2],q[5];
ry(0.7822323108534466) q[2];
ry(-1.713292598477203) q[6];
cx q[2],q[6];
ry(1.226864205389478) q[2];
ry(1.8691662946302707) q[6];
cx q[2],q[6];
ry(-2.9680624896194137) q[2];
ry(0.119025648852217) q[7];
cx q[2],q[7];
ry(2.8688097932099637) q[2];
ry(1.4813445065943607) q[7];
cx q[2],q[7];
ry(-1.2982183091421298) q[2];
ry(-2.8625180796540555) q[8];
cx q[2],q[8];
ry(-1.0842925618788568) q[2];
ry(-0.6357900086657869) q[8];
cx q[2],q[8];
ry(-1.1207493985084085) q[2];
ry(3.0089355262726145) q[9];
cx q[2],q[9];
ry(1.4622897755433681) q[2];
ry(1.4047762563012718) q[9];
cx q[2],q[9];
ry(0.28582894292252714) q[2];
ry(-1.0999480441337939) q[10];
cx q[2],q[10];
ry(-1.689330876272983) q[2];
ry(1.9739533573782244) q[10];
cx q[2],q[10];
ry(-2.173043225725209) q[2];
ry(0.6636514777873367) q[11];
cx q[2],q[11];
ry(-2.198502095331762) q[2];
ry(2.0409006734772364) q[11];
cx q[2],q[11];
ry(-1.4789520217178902) q[3];
ry(-0.2100541994001732) q[4];
cx q[3],q[4];
ry(-1.0620482879673478) q[3];
ry(1.3826597576743225) q[4];
cx q[3],q[4];
ry(0.9154397597486481) q[3];
ry(-0.5741420891964398) q[5];
cx q[3],q[5];
ry(2.6718915264449294) q[3];
ry(-1.8770025489758977) q[5];
cx q[3],q[5];
ry(1.7620939936018916) q[3];
ry(-2.6657224418859276) q[6];
cx q[3],q[6];
ry(1.7560865526750962) q[3];
ry(-1.93964726307245) q[6];
cx q[3],q[6];
ry(1.2549740111316705) q[3];
ry(-0.4194784510800332) q[7];
cx q[3],q[7];
ry(1.7964391803037911) q[3];
ry(-2.6158963035883724) q[7];
cx q[3],q[7];
ry(-2.9407129169509445) q[3];
ry(-0.10981841012556569) q[8];
cx q[3],q[8];
ry(-2.610666950129608) q[3];
ry(-2.170069738912436) q[8];
cx q[3],q[8];
ry(-1.9345184938902265) q[3];
ry(2.569367079945335) q[9];
cx q[3],q[9];
ry(0.04909395213494072) q[3];
ry(-1.4549025232761328) q[9];
cx q[3],q[9];
ry(1.00205833266182) q[3];
ry(-1.140860667677992) q[10];
cx q[3],q[10];
ry(-1.2716249266313209) q[3];
ry(-1.38528876192292) q[10];
cx q[3],q[10];
ry(2.4933123078235804) q[3];
ry(0.5378062460184276) q[11];
cx q[3],q[11];
ry(-1.2238570340382404) q[3];
ry(0.7076254311390385) q[11];
cx q[3],q[11];
ry(-1.5981841056096555) q[4];
ry(2.899309538727205) q[5];
cx q[4],q[5];
ry(-0.21289033626700785) q[4];
ry(0.9956927776169717) q[5];
cx q[4],q[5];
ry(0.5648798836202966) q[4];
ry(0.08451032532664815) q[6];
cx q[4],q[6];
ry(-2.628482098506642) q[4];
ry(-2.6827414751568948) q[6];
cx q[4],q[6];
ry(-2.560902138785694) q[4];
ry(-1.5075889097838657) q[7];
cx q[4],q[7];
ry(2.006416874507976) q[4];
ry(-0.2721279355396673) q[7];
cx q[4],q[7];
ry(-2.654147151955735) q[4];
ry(2.254834531430336) q[8];
cx q[4],q[8];
ry(-1.0515298956166763) q[4];
ry(-1.6952484355197148) q[8];
cx q[4],q[8];
ry(-2.130920862941764) q[4];
ry(2.2760358439490176) q[9];
cx q[4],q[9];
ry(2.5386851190849975) q[4];
ry(1.60731247755728) q[9];
cx q[4],q[9];
ry(0.2639489675986201) q[4];
ry(-1.7003079898856825) q[10];
cx q[4],q[10];
ry(-2.1008501567838858) q[4];
ry(1.512234733057551) q[10];
cx q[4],q[10];
ry(-1.6191398811707238) q[4];
ry(1.6619046575390293) q[11];
cx q[4],q[11];
ry(-1.5504898908612095) q[4];
ry(1.2854380418303446) q[11];
cx q[4],q[11];
ry(-2.744637902615688) q[5];
ry(2.1566282099819185) q[6];
cx q[5],q[6];
ry(0.5987178393284642) q[5];
ry(1.9434885628463063) q[6];
cx q[5],q[6];
ry(-2.8927339749401813) q[5];
ry(1.8216896446219606) q[7];
cx q[5],q[7];
ry(-0.37521830265661116) q[5];
ry(2.4038544567378057) q[7];
cx q[5],q[7];
ry(2.68251226175765) q[5];
ry(-1.423514795712409) q[8];
cx q[5],q[8];
ry(-2.8816753102021244) q[5];
ry(-2.894564516258142) q[8];
cx q[5],q[8];
ry(-1.462427948424354) q[5];
ry(1.0135199992576462) q[9];
cx q[5],q[9];
ry(0.1263122418448368) q[5];
ry(2.2743361620181344) q[9];
cx q[5],q[9];
ry(1.9280987313881344) q[5];
ry(0.7699234211557123) q[10];
cx q[5],q[10];
ry(2.0440975717721943) q[5];
ry(-0.6062710417606709) q[10];
cx q[5],q[10];
ry(0.9211941386344115) q[5];
ry(-1.069204272376452) q[11];
cx q[5],q[11];
ry(0.3386731945667796) q[5];
ry(0.724147358172423) q[11];
cx q[5],q[11];
ry(-1.1365777133289652) q[6];
ry(1.9478000505693194) q[7];
cx q[6],q[7];
ry(2.823681731942819) q[6];
ry(1.0090213833083534) q[7];
cx q[6],q[7];
ry(2.4401148635345113) q[6];
ry(-0.7210767172466079) q[8];
cx q[6],q[8];
ry(-1.765245904839607) q[6];
ry(2.5039689968356242) q[8];
cx q[6],q[8];
ry(0.10927850660273553) q[6];
ry(-0.08367549447979085) q[9];
cx q[6],q[9];
ry(2.8987359506774624) q[6];
ry(-0.4569213508132419) q[9];
cx q[6],q[9];
ry(-1.049648122877103) q[6];
ry(-1.307637095078146) q[10];
cx q[6],q[10];
ry(-0.4491797596138376) q[6];
ry(2.6528645239592294) q[10];
cx q[6],q[10];
ry(0.16545227811884677) q[6];
ry(1.2205826653496668) q[11];
cx q[6],q[11];
ry(1.5804808552023148) q[6];
ry(-1.854336979417707) q[11];
cx q[6],q[11];
ry(1.1367784407571717) q[7];
ry(1.720083764108504) q[8];
cx q[7],q[8];
ry(2.7664739944763) q[7];
ry(1.1509144347888296) q[8];
cx q[7],q[8];
ry(0.16822473228320606) q[7];
ry(-0.603345806068774) q[9];
cx q[7],q[9];
ry(1.0285629599507287) q[7];
ry(-1.6632442439201447) q[9];
cx q[7],q[9];
ry(-0.025411210956357178) q[7];
ry(-1.2743115913611165) q[10];
cx q[7],q[10];
ry(-2.287140325689685) q[7];
ry(-1.2533598432500008) q[10];
cx q[7],q[10];
ry(1.1649788493834174) q[7];
ry(2.6668133616742735) q[11];
cx q[7],q[11];
ry(-2.8913809090164424) q[7];
ry(-2.453964982152702) q[11];
cx q[7],q[11];
ry(1.4929046976829126) q[8];
ry(1.2303256455998977) q[9];
cx q[8],q[9];
ry(2.276292366141857) q[8];
ry(-1.83051647070549) q[9];
cx q[8],q[9];
ry(-0.9077078624898908) q[8];
ry(-0.13276959847567937) q[10];
cx q[8],q[10];
ry(0.8387646106668274) q[8];
ry(0.6660061328755207) q[10];
cx q[8],q[10];
ry(2.0037648705296203) q[8];
ry(0.7661214367963387) q[11];
cx q[8],q[11];
ry(-0.4896049784683771) q[8];
ry(0.8055497958648354) q[11];
cx q[8],q[11];
ry(-0.33504541355162587) q[9];
ry(1.4919316715167772) q[10];
cx q[9],q[10];
ry(0.12407561099778608) q[9];
ry(0.9996111207166438) q[10];
cx q[9],q[10];
ry(0.7903732711082663) q[9];
ry(2.5853591584350353) q[11];
cx q[9],q[11];
ry(-0.27154698785300174) q[9];
ry(0.5276233382982358) q[11];
cx q[9],q[11];
ry(-1.0320920874452535) q[10];
ry(2.471115262206008) q[11];
cx q[10],q[11];
ry(0.45783978361840594) q[10];
ry(0.6481542056848131) q[11];
cx q[10],q[11];
ry(0.46202221591202) q[0];
ry(-2.8787974560893033) q[1];
ry(-2.889913538985658) q[2];
ry(0.31491485694570254) q[3];
ry(-1.540782316092959) q[4];
ry(2.2695803622831) q[5];
ry(-0.3263350562647337) q[6];
ry(0.3521362027260782) q[7];
ry(2.669235178957428) q[8];
ry(0.9363827855755318) q[9];
ry(-0.2249535314999452) q[10];
ry(0.7761584987261125) q[11];