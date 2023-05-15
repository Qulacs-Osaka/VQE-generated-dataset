OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.942414943789439) q[0];
rz(-0.0821569119787462) q[0];
ry(2.0963558435752945) q[1];
rz(-2.003910971449361) q[1];
ry(-0.3374646005899571) q[2];
rz(-1.064131459545459) q[2];
ry(3.137035017896753) q[3];
rz(-2.051964660419865) q[3];
ry(-3.135071391575645) q[4];
rz(-1.6245904464752838) q[4];
ry(-2.783874624693603) q[5];
rz(0.7586965807747172) q[5];
ry(2.819215312191509) q[6];
rz(1.1400896592622265) q[6];
ry(2.5858196128438835) q[7];
rz(-1.221881293834878) q[7];
ry(-3.1386325044491605) q[8];
rz(-0.49773242777212273) q[8];
ry(-0.011137175032529529) q[9];
rz(0.21788133537890794) q[9];
ry(-1.1520764866418312) q[10];
rz(0.4466295204913848) q[10];
ry(3.1360976761653574) q[11];
rz(-0.7971670897433007) q[11];
ry(-0.0007255027003632009) q[12];
rz(2.4746576843552766) q[12];
ry(-1.8364569397527348) q[13];
rz(-0.6223399805308043) q[13];
ry(-0.38298451050970733) q[14];
rz(1.095401010814971) q[14];
ry(-0.0022885411354256007) q[15];
rz(-0.23186456528059196) q[15];
ry(-0.31536643183766705) q[16];
rz(0.6206498008516794) q[16];
ry(0.1793632381613195) q[17];
rz(0.12652917696061472) q[17];
ry(1.7436724534237467) q[18];
rz(-1.1531733835499105) q[18];
ry(-1.095952866661619) q[19];
rz(-2.8344387709036694) q[19];
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
ry(-0.07121138793604159) q[0];
rz(-0.7809889707117293) q[0];
ry(2.543001697332169) q[1];
rz(1.519852923049302) q[1];
ry(1.4702152392283132) q[2];
rz(-0.4770037943676843) q[2];
ry(-3.138191720875284) q[3];
rz(-1.7442818171175605) q[3];
ry(-0.0007168821010212767) q[4];
rz(-0.9704567895058571) q[4];
ry(-3.1386039981385476) q[5];
rz(-2.2429757267296706) q[5];
ry(3.1356593356970817) q[6];
rz(-0.651979694672078) q[6];
ry(0.01286389454158865) q[7];
rz(-2.130690222859897) q[7];
ry(0.0013975427412636601) q[8];
rz(2.1887473958384702) q[8];
ry(-0.014846414546527902) q[9];
rz(1.165196640515605) q[9];
ry(-2.104752217695817) q[10];
rz(-0.6242195135956168) q[10];
ry(-0.10222491124587502) q[11];
rz(1.01893105282175) q[11];
ry(-3.1410644115252087) q[12];
rz(1.2972279361988972) q[12];
ry(1.1386676684025998) q[13];
rz(3.045334159356781) q[13];
ry(1.8609838652211943) q[14];
rz(2.935780350309511) q[14];
ry(-0.002259615121010583) q[15];
rz(2.086832328187254) q[15];
ry(-1.4030217351423075) q[16];
rz(-1.2181246463994608) q[16];
ry(-1.3950912420212531) q[17];
rz(-1.3144895783294364) q[17];
ry(-2.044447164604324) q[18];
rz(-0.7388517703090791) q[18];
ry(-0.9576630757003077) q[19];
rz(2.46229015823814) q[19];
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
ry(2.9773075412857057) q[0];
rz(0.9693982310572544) q[0];
ry(0.8603521254761066) q[1];
rz(2.4110242317489674) q[1];
ry(-0.1746794926649231) q[2];
rz(-1.7314858019049293) q[2];
ry(0.009567098381266197) q[3];
rz(-2.4738697518102413) q[3];
ry(0.47615701601113813) q[4];
rz(1.8974350865409975) q[4];
ry(2.6647286440928686) q[5];
rz(-2.372041321076718) q[5];
ry(2.6616428071275857) q[6];
rz(-0.8509429010692582) q[6];
ry(2.653687596083476) q[7];
rz(2.2322719917070977) q[7];
ry(0.0006772503624321429) q[8];
rz(2.2775534832178015) q[8];
ry(3.1372538868531206) q[9];
rz(-0.5654195669565283) q[9];
ry(-1.3933028100574454) q[10];
rz(2.588990915244305) q[10];
ry(-1.5348490580333785) q[11];
rz(-1.449335860799298) q[11];
ry(0.00031176933318022293) q[12];
rz(-0.9541278728076499) q[12];
ry(1.767461968074192) q[13];
rz(-2.5252397996364984) q[13];
ry(-0.5215349527822193) q[14];
rz(0.9450328269895564) q[14];
ry(3.140645486789286) q[15];
rz(-0.9494083791672322) q[15];
ry(-1.5181783058763294) q[16];
rz(0.03946540814019351) q[16];
ry(2.606684979693608) q[17];
rz(1.5434866666906553) q[17];
ry(1.6471314978328055) q[18];
rz(-1.6203460834804788) q[18];
ry(-1.0692356647219663) q[19];
rz(-2.3844285362709026) q[19];
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
ry(-0.8449110170519676) q[0];
rz(2.7420719503730258) q[0];
ry(1.3957047553936863) q[1];
rz(0.04611686069916862) q[1];
ry(-3.0992467452387724) q[2];
rz(2.7793546600655246) q[2];
ry(-3.1289306131004646) q[3];
rz(0.8737992909201724) q[3];
ry(3.1379034325052597) q[4];
rz(-2.1784889364768194) q[4];
ry(0.23972461310197182) q[5];
rz(-1.4136232582354902) q[5];
ry(-3.047128871249555) q[6];
rz(2.949178121888446) q[6];
ry(-3.135444198922028) q[7];
rz(2.9257472068554145) q[7];
ry(-1.572916610936251) q[8];
rz(-0.5888331839398371) q[8];
ry(3.1409349520165044) q[9];
rz(3.005402421357086) q[9];
ry(-1.6027724120517337) q[10];
rz(-1.5339039159554322) q[10];
ry(-2.2806541214363953) q[11];
rz(-1.2981976666223898) q[11];
ry(-1.5961797016335746) q[12];
rz(-0.25313341112531074) q[12];
ry(1.4945956826824291) q[13];
rz(1.6135397659172916) q[13];
ry(3.0798403027556014) q[14];
rz(1.9932146306640188) q[14];
ry(1.5614123792376287) q[15];
rz(1.3885450446557475) q[15];
ry(1.9893933827500718) q[16];
rz(2.880222296294486) q[16];
ry(-1.1822174415143927) q[17];
rz(-3.112176840610781) q[17];
ry(0.008655524650822012) q[18];
rz(2.5911343512309104) q[18];
ry(0.20255776971183584) q[19];
rz(-1.5824115305555706) q[19];
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
ry(-1.5097417942820446) q[0];
rz(-2.8327969448163413) q[0];
ry(-0.8653091272258422) q[1];
rz(-3.1286085738101783) q[1];
ry(0.1751096531225249) q[2];
rz(-2.746765546333573) q[2];
ry(3.1413626407624453) q[3];
rz(-0.7331394732911576) q[3];
ry(0.051628127164478954) q[4];
rz(2.9184469269327) q[4];
ry(-2.861130327615514) q[5];
rz(0.2542229976703787) q[5];
ry(1.572553176909819) q[6];
rz(-2.654124126112974) q[6];
ry(-1.6532385473787168) q[7];
rz(-2.3056596996308367) q[7];
ry(-0.0013198226350468309) q[8];
rz(-2.157904476529093) q[8];
ry(1.5685201152953254) q[9];
rz(-3.071286857956584) q[9];
ry(-0.002596048301263032) q[10];
rz(1.6646007488577004) q[10];
ry(-1.5448119479510645) q[11];
rz(0.7866246484534873) q[11];
ry(0.00021243050541208672) q[12];
rz(-1.3166958427602058) q[12];
ry(2.8370587776385956) q[13];
rz(-0.2890788111283155) q[13];
ry(3.141540400085265) q[14];
rz(0.08983338186924517) q[14];
ry(-3.141432784306569) q[15];
rz(1.4355358988694134) q[15];
ry(2.0864615434008416) q[16];
rz(-1.6065138757044186) q[16];
ry(0.00011868061458682733) q[17];
rz(-0.0743724172402244) q[17];
ry(0.2431042867877383) q[18];
rz(-2.698243319866553) q[18];
ry(0.2752542163576024) q[19];
rz(-2.3233420285670525) q[19];
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
ry(-0.9884353757110534) q[0];
rz(-0.534504800746937) q[0];
ry(1.3525702823860624) q[1];
rz(2.4954231512719955) q[1];
ry(1.611691272647324) q[2];
rz(-0.24935230643617154) q[2];
ry(0.0009940094736166413) q[3];
rz(2.092684783549477) q[3];
ry(-2.1137705009059116) q[4];
rz(0.903414174791272) q[4];
ry(-0.6766510451832364) q[5];
rz(0.08675227517909523) q[5];
ry(-3.1252638278267115) q[6];
rz(2.691933205675421) q[6];
ry(-3.1403559828518617) q[7];
rz(-1.3121467927309958) q[7];
ry(-2.5243472966208182) q[8];
rz(-0.5296248049200248) q[8];
ry(3.1350373078162646) q[9];
rz(0.24342016931891933) q[9];
ry(-2.9932222462006814) q[10];
rz(1.9631257700770262) q[10];
ry(1.5709559326119928) q[11];
rz(-0.0004151987899936315) q[11];
ry(1.5693363052856037) q[12];
rz(2.8539260710827232) q[12];
ry(-3.095720704845036) q[13];
rz(-1.8643100799122143) q[13];
ry(-0.7625499119515737) q[14];
rz(2.5610470229783204) q[14];
ry(0.02604462643861627) q[15];
rz(3.059589922724439) q[15];
ry(-1.6991367917707592) q[16];
rz(0.9422232628906348) q[16];
ry(-1.6423548091830003) q[17];
rz(2.2938317899731064) q[17];
ry(0.45321723419397036) q[18];
rz(0.006513463093586874) q[18];
ry(1.3313642727623856) q[19];
rz(1.7677057565336645) q[19];
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
ry(1.551312344333189) q[0];
rz(0.8246477590138249) q[0];
ry(0.6037116455102136) q[1];
rz(0.7128119863965822) q[1];
ry(3.1303042439712) q[2];
rz(2.77117581032849) q[2];
ry(-3.107017184445779) q[3];
rz(2.678400685306777) q[3];
ry(-0.0018986709236568087) q[4];
rz(-2.5407223639508274) q[4];
ry(2.733709365790531) q[5];
rz(-0.10351354068047362) q[5];
ry(-3.1401620801325545) q[6];
rz(-2.8870577546982714) q[6];
ry(-0.1927340293340176) q[7];
rz(-0.18371142356836687) q[7];
ry(0.000923615146278145) q[8];
rz(3.072812123580252) q[8];
ry(-0.011071993488148711) q[9];
rz(2.3422852533813066) q[9];
ry(0.004851925517665556) q[10];
rz(-1.6322838185648623) q[10];
ry(1.7399002940625579) q[11];
rz(1.6042320690574368) q[11];
ry(0.0009423374375430172) q[12];
rz(-3.099693156261976) q[12];
ry(-1.5700211945828673) q[13];
rz(-3.1415757749381967) q[13];
ry(-9.074638628181475e-05) q[14];
rz(1.405807094939279) q[14];
ry(0.0009684304992991182) q[15];
rz(0.0356847785967441) q[15];
ry(1.176718308248323) q[16];
rz(-1.847186177189843) q[16];
ry(3.1415215936206193) q[17];
rz(2.399557456067169) q[17];
ry(-1.2928446393969866) q[18];
rz(-0.7583919438715601) q[18];
ry(-1.3838950438255893) q[19];
rz(-0.0016572476467506733) q[19];
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
ry(-0.5854555726733723) q[0];
rz(0.8292954760172643) q[0];
ry(-3.1178471570967425) q[1];
rz(-1.711379860988184) q[1];
ry(2.5152257731544423) q[2];
rz(-0.9073372450950352) q[2];
ry(3.134082490302023) q[3];
rz(-2.748305370797786) q[3];
ry(2.022595175718638) q[4];
rz(-1.7338277658678631) q[4];
ry(2.5622771602022794) q[5];
rz(-0.10170894577868062) q[5];
ry(3.1343382684395884) q[6];
rz(-0.24565609660776214) q[6];
ry(3.1307883899719755) q[7];
rz(2.3125017355441826) q[7];
ry(-2.6972723633031364) q[8];
rz(1.346565325151266) q[8];
ry(-9.210336169651612e-05) q[9];
rz(1.3954395066479872) q[9];
ry(1.5324029101364092) q[10];
rz(-2.64368515663569) q[10];
ry(3.1393356283142317) q[11];
rz(-1.537115049865874) q[11];
ry(-1.843503122897852) q[12];
rz(1.9618541050562852) q[12];
ry(1.5726607364880512) q[13];
rz(-3.141533394746324) q[13];
ry(-0.5452602960434394) q[14];
rz(2.966732488435483) q[14];
ry(-1.5707434077759759) q[15];
rz(1.3417170935648617) q[15];
ry(-1.4682583653497403) q[16];
rz(-2.4311403373472062) q[16];
ry(-2.602758353634717) q[17];
rz(-1.1413148859758317) q[17];
ry(0.8561968181277253) q[18];
rz(0.486644748973066) q[18];
ry(2.8246272047797287) q[19];
rz(1.1339402086902746) q[19];
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
ry(1.196535153288691) q[0];
rz(0.8948193265252585) q[0];
ry(-3.1394549080370773) q[1];
rz(-0.27992572729411874) q[1];
ry(-1.5740372051742906) q[2];
rz(-1.9971467451718317) q[2];
ry(1.586374552365923) q[3];
rz(0.010708632532645715) q[3];
ry(3.1395945527847977) q[4];
rz(-0.6899657653412232) q[4];
ry(3.0122724965357794) q[5];
rz(-1.5934964677367036) q[5];
ry(3.141039536794113) q[6];
rz(-2.010156908357371) q[6];
ry(0.5603820261942474) q[7];
rz(1.575651134157023) q[7];
ry(3.141301384582956) q[8];
rz(-2.560434967393274) q[8];
ry(0.011341530361931782) q[9];
rz(1.0013859786340062) q[9];
ry(-3.1412940030669128) q[10];
rz(0.38632181286406664) q[10];
ry(-1.5735958667189154) q[11];
rz(1.5424776455490143) q[11];
ry(3.1411380653488483) q[12];
rz(1.996381494730516) q[12];
ry(-1.5750621521507684) q[13];
rz(0.06720348156525484) q[13];
ry(3.141429275818769) q[14];
rz(1.2431694446524661) q[14];
ry(3.1408375193342564) q[15];
rz(1.6242240556595577) q[15];
ry(-3.073388164627718) q[16];
rz(-0.6025264307049651) q[16];
ry(0.2845600141715754) q[17];
rz(3.139577622538281) q[17];
ry(-0.18429371990416413) q[18];
rz(-0.9316111644132256) q[18];
ry(1.3407615220107179) q[19];
rz(1.3271836988714074) q[19];
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
ry(1.571118892922036) q[0];
rz(-1.5702718055748621) q[0];
ry(-0.0005454470346544227) q[1];
rz(1.2123936674103133) q[1];
ry(-0.001366654097809672) q[2];
rz(1.9956986361056133) q[2];
ry(1.5712039558142017) q[3];
rz(1.0809381304273584) q[3];
ry(3.135558187835586) q[4];
rz(3.0943761398579466) q[4];
ry(1.554582285203086) q[5];
rz(1.8976196616762866) q[5];
ry(3.140737408587264) q[6];
rz(0.9698112556248893) q[6];
ry(-1.5779945020823194) q[7];
rz(1.7285896533756657) q[7];
ry(-1.0041991356282427) q[8];
rz(2.375079779766446) q[8];
ry(3.141027728924966) q[9];
rz(0.09230650479488772) q[9];
ry(1.5836455413069832) q[10];
rz(3.083527770420667) q[10];
ry(1.5578732363286072) q[11];
rz(1.5890885952932219) q[11];
ry(0.020758879449139123) q[12];
rz(-2.874558877121422) q[12];
ry(-1.2649192178556827) q[13];
rz(1.3416060936179175) q[13];
ry(-0.7867250179755211) q[14];
rz(2.944595117123278) q[14];
ry(-1.5636745087293304) q[15];
rz(-2.930763783471531) q[15];
ry(-1.569299335677658) q[16];
rz(1.571652082069796) q[16];
ry(1.3800448198369224) q[17];
rz(-1.6027093198170816) q[17];
ry(1.2225214052491) q[18];
rz(-3.08297756824632) q[18];
ry(-3.130060814320012) q[19];
rz(-2.519281193374011) q[19];
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
ry(-1.5728130425525721) q[0];
rz(2.2883854394349643) q[0];
ry(-0.001380269377637947) q[1];
rz(2.5018304358273347) q[1];
ry(1.5657152515137895) q[2];
rz(-3.1394127627135) q[2];
ry(1.4683712480987179) q[3];
rz(-1.7610824139895214) q[3];
ry(3.109834025484477) q[4];
rz(-0.02204110198309461) q[4];
ry(3.1415515716384843) q[5];
rz(-1.0534317076438688) q[5];
ry(3.0895149683012235) q[6];
rz(2.6289220086761507) q[6];
ry(0.00013323916388766597) q[7];
rz(-2.710994979139103) q[7];
ry(-0.16085468099276223) q[8];
rz(3.028081733904104) q[8];
ry(2.9523057613528874) q[9];
rz(0.07965970669888199) q[9];
ry(-2.451402237258108) q[10];
rz(2.5064090288471763) q[10];
ry(1.5731656103626737) q[11];
rz(3.1405433244969077) q[11];
ry(0.01997131179141931) q[12];
rz(0.7869595871953725) q[12];
ry(3.103143854580727) q[13];
rz(-1.816216718071801) q[13];
ry(-2.95825598302012) q[14];
rz(-0.43840212502604553) q[14];
ry(0.23422739523701563) q[15];
rz(2.452780891562362) q[15];
ry(-1.5738364797816082) q[16];
rz(1.441026653909926) q[16];
ry(-3.0513111691499635) q[17];
rz(3.1078726866912336) q[17];
ry(-1.5585852913364455) q[18];
rz(1.7491914122218108) q[18];
ry(-0.17557823171330955) q[19];
rz(2.5670983475715343) q[19];
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
ry(3.1096068729794366) q[0];
rz(-1.7417065103275984) q[0];
ry(-0.020076524818715136) q[1];
rz(3.1194492777828717) q[1];
ry(-1.5705405293341483) q[2];
rz(-1.5703496151258003) q[2];
ry(1.5706660290865526) q[3];
rz(-3.0002274762291177) q[3];
ry(1.5710216472606346) q[4];
rz(2.264285849072983) q[4];
ry(1.3326973940952436) q[5];
rz(2.972125007974603) q[5];
ry(-3.1402651952721268) q[6];
rz(0.4123931384280706) q[6];
ry(-0.15786205735666847) q[7];
rz(-2.946553133871595) q[7];
ry(-3.0086028218401304) q[8];
rz(1.7238388857453568) q[8];
ry(1.1467642157731426e-05) q[9];
rz(3.076548403690709) q[9];
ry(3.141443811078226) q[10];
rz(-0.25447219333058424) q[10];
ry(1.5190612956392242) q[11];
rz(0.6252671729142298) q[11];
ry(-0.0001962123169594321) q[12];
rz(-1.0272980651441026) q[12];
ry(0.0595365116539021) q[13];
rz(1.6067538261697347) q[13];
ry(0.0020610564787303938) q[14];
rz(-1.6186578193001155) q[14];
ry(0.0045436094479320835) q[15];
rz(-1.9402308218835813) q[15];
ry(1.5776731351294666) q[16];
rz(-0.044804073789818176) q[16];
ry(1.6406866536334936) q[17];
rz(-1.570115121000582) q[17];
ry(-3.0899536964777585) q[18];
rz(0.33866578194628205) q[18];
ry(-1.6432874132776751) q[19];
rz(1.7057574610420936) q[19];
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
ry(-0.06677055213442529) q[0];
rz(1.5773008665626789) q[0];
ry(0.025981411366505926) q[1];
rz(-1.5799289544792623) q[1];
ry(-1.5826006184065884) q[2];
rz(1.1759418094688332) q[2];
ry(3.132495345189745) q[3];
rz(-1.273477954940037) q[3];
ry(-0.0008732672202969027) q[4];
rz(-0.5251437447023112) q[4];
ry(3.139880954356216) q[5];
rz(-2.6313026076269823) q[5];
ry(0.04495008265023248) q[6];
rz(-1.397063620324863) q[6];
ry(-3.1405601631308224) q[7];
rz(2.3391614122861095) q[7];
ry(-1.7065211725670204) q[8];
rz(-1.9569686903075485) q[8];
ry(-0.019304596193209456) q[9];
rz(3.069312654113771) q[9];
ry(0.022779730129144582) q[10];
rz(1.2043093555755389) q[10];
ry(0.044977959229044746) q[11];
rz(2.5159029665223227) q[11];
ry(1.5939139785562524) q[12];
rz(-0.004627104039249953) q[12];
ry(-1.5549851540147817) q[13];
rz(-1.0037084176199245) q[13];
ry(-3.141349793104804) q[14];
rz(1.0830854050907506) q[14];
ry(-3.1338944783832536) q[15];
rz(3.071334763288141) q[15];
ry(-1.5686144530366652) q[16];
rz(3.1335344830435994) q[16];
ry(1.5737700834017665) q[17];
rz(-0.1342186081492116) q[17];
ry(1.5910181900404494) q[18];
rz(2.9978644360260613) q[18];
ry(-3.1401159325566894) q[19];
rz(-1.4928373383493325) q[19];
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
ry(1.5717804547661869) q[0];
rz(0.48029939405486743) q[0];
ry(0.020441539023697608) q[1];
rz(-0.297662568513241) q[1];
ry(-3.141222104438293) q[2];
rz(2.740867565821748) q[2];
ry(8.10993783222358e-05) q[3];
rz(1.1233400080813958) q[3];
ry(-3.1392375080945074) q[4];
rz(-1.0665008632149433) q[4];
ry(-2.839608630070276) q[5];
rz(-0.9293477032892286) q[5];
ry(-0.0025608784454611566) q[6];
rz(0.9368070603130025) q[6];
ry(-1.6575941567557806) q[7];
rz(0.2790312159294445) q[7];
ry(-1.5637495253554272) q[8];
rz(1.5800254565557261) q[8];
ry(-0.3293366474521991) q[9];
rz(3.116868572768027) q[9];
ry(-2.9005478573706194) q[10];
rz(-1.9624131497946968) q[10];
ry(-0.7474687700154803) q[11];
rz(3.0534715953901537) q[11];
ry(2.8279277840180748) q[12];
rz(-3.110154869460697) q[12];
ry(0.0001642905496019864) q[13];
rz(1.0030450166445606) q[13];
ry(1.5922463968294158) q[14];
rz(-0.018066099347956815) q[14];
ry(-3.1415429781522546) q[15];
rz(0.9860304028352699) q[15];
ry(-1.524624532978236) q[16];
rz(0.08331843061755816) q[16];
ry(0.0025890525620647464) q[17];
rz(-3.025155236111147) q[17];
ry(1.5687687272636965) q[18];
rz(-1.5731122688970443) q[18];
ry(-0.006271675082724748) q[19];
rz(-0.8717095703348771) q[19];
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
ry(-1.6357508512878713) q[0];
rz(-1.5436386470879861) q[0];
ry(-1.5723668950990333) q[1];
rz(-0.16960152238964543) q[1];
ry(1.6078816884641594) q[2];
rz(1.747141812675473) q[2];
ry(2.512965160703287) q[3];
rz(1.0225685323164067) q[3];
ry(-1.6992607141111522) q[4];
rz(1.4555586121186386) q[4];
ry(-1.510001085476141) q[5];
rz(-1.5299243695194538) q[5];
ry(-1.564233691329484) q[6];
rz(-2.548749062492987) q[6];
ry(-3.133368209554887) q[7];
rz(1.7780568782347697) q[7];
ry(-3.0018279707077125) q[8];
rz(-1.5874210614789561) q[8];
ry(3.104481861006808) q[9];
rz(3.1170102574812653) q[9];
ry(0.0012933073922243164) q[10];
rz(1.969765248455281) q[10];
ry(-3.1216613671648505) q[11];
rz(-0.08775764964937771) q[11];
ry(0.0006150306476060409) q[12];
rz(-0.026902444116823073) q[12];
ry(-1.635616727177929) q[13];
rz(2.958825817000573) q[13];
ry(3.0801388720058536) q[14];
rz(-1.58758115828112) q[14];
ry(2.9197833793294885) q[15];
rz(-2.9939783074609627) q[15];
ry(-3.133559554527121) q[16];
rz(0.08347922583587586) q[16];
ry(3.0870718245905184) q[17];
rz(-1.6083287041793997) q[17];
ry(1.572321061768804) q[18];
rz(1.159749719041302) q[18];
ry(-0.002233077750708589) q[19];
rz(-2.2051763118178735) q[19];
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
ry(1.5716829673372557) q[0];
rz(0.00482212403791582) q[0];
ry(-0.02376950641635389) q[1];
rz(1.0178674049024075) q[1];
ry(-3.1398951040175964) q[2];
rz(-3.09395773845126) q[2];
ry(-0.00014775715438583847) q[3];
rz(0.2733646308729575) q[3];
ry(3.1409093938026813) q[4];
rz(-2.319667618650369) q[4];
ry(3.1410490652796064) q[5];
rz(-2.133057194018643) q[5];
ry(3.1385055525879397) q[6];
rz(-0.9574313711651703) q[6];
ry(-1.5683190776759606) q[7];
rz(0.6119307881811755) q[7];
ry(-1.5747695885694986) q[8];
rz(-2.3774527919294286) q[8];
ry(2.8150829457598374) q[9];
rz(1.6310680474683847) q[9];
ry(2.9030803532307328) q[10];
rz(0.0709383657060689) q[10];
ry(1.6186164221467958) q[11];
rz(0.06083212140804407) q[11];
ry(3.135825566821408) q[12];
rz(2.960469014721646) q[12];
ry(2.5277016678822437) q[13];
rz(3.1262598240345105) q[13];
ry(-1.568838965132814) q[14];
rz(-0.2963532392728127) q[14];
ry(3.1405666408811497) q[15];
rz(0.793047461428094) q[15];
ry(1.5600825970049144) q[16];
rz(1.077473668201936) q[16];
ry(-1.5581847226489813) q[17];
rz(0.9920967095562893) q[17];
ry(0.3320702632734873) q[18];
rz(0.48085231295534475) q[18];
ry(0.13603171269986714) q[19];
rz(-2.981728520192687) q[19];
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
ry(0.47760686549313436) q[0];
rz(-3.015717534978981) q[0];
ry(1.5857689345797494) q[1];
rz(1.6045929763121487) q[1];
ry(-0.6459068474950298) q[2];
rz(2.332565340719745) q[2];
ry(-1.3651821818942036) q[3];
rz(0.9425143140591145) q[3];
ry(2.1406173618732396) q[4];
rz(-1.7195721588772406) q[4];
ry(3.1412795384123746) q[5];
rz(2.286654119298486) q[5];
ry(-3.092566184853585) q[6];
rz(0.020530506536397094) q[6];
ry(0.0004003278814899147) q[7];
rz(2.5297319076914886) q[7];
ry(1.3310017505906613) q[8];
rz(1.3468402072225993) q[8];
ry(0.20236526485327463) q[9];
rz(1.8965722771661249) q[9];
ry(1.5709265950028994) q[10];
rz(-1.627395400450957) q[10];
ry(-0.0005590238742045538) q[11];
rz(-3.0310768824553844) q[11];
ry(-2.1396633381195497e-05) q[12];
rz(0.8424237626195933) q[12];
ry(-2.9788278998472295) q[13];
rz(3.127903039989432) q[13];
ry(-0.31324573725027527) q[14];
rz(-0.7171585101169475) q[14];
ry(-3.140846253572823) q[15];
rz(-2.2696664850394646) q[15];
ry(3.1307776753157173) q[16];
rz(-0.8478316169536002) q[16];
ry(-2.700340141958903) q[17];
rz(0.052510879636372555) q[17];
ry(1.5025774274781665) q[18];
rz(2.780364788631756) q[18];
ry(0.11693356248028319) q[19];
rz(0.5971461833364389) q[19];
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
ry(0.0008714657705528595) q[0];
rz(-2.522628838052866) q[0];
ry(1.5610563924532093) q[1];
rz(0.38714248466289425) q[1];
ry(-3.141357221726457) q[2];
rz(-2.332323587409538) q[2];
ry(3.1410697422005036) q[3];
rz(1.2577562293802842) q[3];
ry(-3.1406589753768555) q[4];
rz(-0.8144546303811547) q[4];
ry(-0.00026280563924338816) q[5];
rz(-0.1165301148939557) q[5];
ry(0.0006593678476427556) q[6];
rz(2.0797434552499547) q[6];
ry(-1.5684489733655962) q[7];
rz(2.1108037981767094) q[7];
ry(1.5712000041778822) q[8];
rz(0.6676192972676933) q[8];
ry(0.0006015168102106415) q[9];
rz(-0.48129127607796374) q[9];
ry(-0.0003597911158381706) q[10];
rz(-2.008597647628795) q[10];
ry(-3.9778696830679866e-05) q[11];
rz(1.3746751469605019) q[11];
ry(9.77297652233586e-05) q[12];
rz(-2.046632982748494) q[12];
ry(-0.6140076751011104) q[13];
rz(-2.977781088668625) q[13];
ry(-0.0012615766601898881) q[14];
rz(0.005480199108276906) q[14];
ry(-0.0002841783973153156) q[15];
rz(1.6112136628285656) q[15];
ry(-3.1412774935179253) q[16];
rz(1.0235325960760162) q[16];
ry(-0.023913032959439917) q[17];
rz(1.888226505954362) q[17];
ry(2.8228011110878555) q[18];
rz(-1.7400274558978799) q[18];
ry(0.003095569080985727) q[19];
rz(1.2476767973445062) q[19];
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
ry(2.0366429750952033) q[0];
rz(-1.2235392368338707) q[0];
ry(1.2579618632685952) q[1];
rz(-1.4172725040181735) q[1];
ry(0.5300083273120975) q[2];
rz(-1.365925882077348) q[2];
ry(0.37775523515093723) q[3];
rz(-0.832827459024962) q[3];
ry(-1.4843281970245243) q[4];
rz(0.09290893521471608) q[4];
ry(-1.5120473798068454) q[5];
rz(0.4285341551800989) q[5];
ry(-0.5571348866836621) q[6];
rz(1.3312718639921315) q[6];
ry(1.5647537111193364) q[7];
rz(0.42483735101708914) q[7];
ry(-2.7208966025704555) q[8];
rz(2.3983390278266645) q[8];
ry(0.5507624342152715) q[9];
rz(-0.9959464372553953) q[9];
ry(-1.8266386956853597) q[10];
rz(0.20431770283565817) q[10];
ry(-0.5381157226982791) q[11];
rz(2.122650069664836) q[11];
ry(-1.0221782428097899) q[12];
rz(-1.296857385723012) q[12];
ry(-3.0912513335944074) q[13];
rz(2.095698259663858) q[13];
ry(2.3595179734168816) q[14];
rz(2.557029338388172) q[14];
ry(2.561000359938503) q[15];
rz(2.1501711610185694) q[15];
ry(2.122148132989829) q[16];
rz(-1.3146605607702686) q[16];
ry(2.5471466266104663) q[17];
rz(2.9555805703501776) q[17];
ry(-1.0551868777173903) q[18];
rz(-1.187261714580325) q[18];
ry(2.539524463298369) q[19];
rz(2.3411705026863143) q[19];