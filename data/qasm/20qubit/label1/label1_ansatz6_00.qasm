OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.9365265199316797) q[0];
ry(0.8543466065865511) q[1];
cx q[0],q[1];
ry(-2.969092283037415) q[0];
ry(-1.9726878321862076) q[1];
cx q[0],q[1];
ry(-1.755923780532143) q[1];
ry(-2.9495616647949854) q[2];
cx q[1],q[2];
ry(-1.321967518072526) q[1];
ry(-2.507076424145449) q[2];
cx q[1],q[2];
ry(2.7023400737873455) q[2];
ry(-1.4451928997035743) q[3];
cx q[2],q[3];
ry(-0.11991392371533749) q[2];
ry(-1.3968689092518947) q[3];
cx q[2],q[3];
ry(1.3117325157358257) q[3];
ry(0.360388199801867) q[4];
cx q[3],q[4];
ry(0.7350602902426989) q[3];
ry(0.6401346894861949) q[4];
cx q[3],q[4];
ry(2.5957623003577286) q[4];
ry(-0.9561570797969795) q[5];
cx q[4],q[5];
ry(2.890950690398228) q[4];
ry(-2.8972748460089464) q[5];
cx q[4],q[5];
ry(-0.15367699730860895) q[5];
ry(-1.8805382979572574) q[6];
cx q[5],q[6];
ry(1.8327533441823762) q[5];
ry(2.2752367603766874) q[6];
cx q[5],q[6];
ry(1.5885844968895704) q[6];
ry(-0.17198036291185775) q[7];
cx q[6],q[7];
ry(-1.4285781675458737) q[6];
ry(-0.28550114703876606) q[7];
cx q[6],q[7];
ry(0.8692563298524104) q[7];
ry(-1.3188221213369544) q[8];
cx q[7],q[8];
ry(2.0802155279444356) q[7];
ry(-2.8615105323803633) q[8];
cx q[7],q[8];
ry(-2.5339771505073423) q[8];
ry(-2.018178352253671) q[9];
cx q[8],q[9];
ry(1.804768962552147) q[8];
ry(0.8236203116632845) q[9];
cx q[8],q[9];
ry(-2.6957440652470446) q[9];
ry(2.7880756435715317) q[10];
cx q[9],q[10];
ry(-1.4142509984045484) q[9];
ry(-2.966721996035603) q[10];
cx q[9],q[10];
ry(1.8734081203024362) q[10];
ry(-1.0991123424795088) q[11];
cx q[10],q[11];
ry(2.863227558569609) q[10];
ry(2.826516715897759) q[11];
cx q[10],q[11];
ry(2.87578754431052) q[11];
ry(0.8009601309753799) q[12];
cx q[11],q[12];
ry(2.247854538198085) q[11];
ry(1.3148171048241284) q[12];
cx q[11],q[12];
ry(1.2687810812869929) q[12];
ry(-2.5594556332307454) q[13];
cx q[12],q[13];
ry(-1.7186527768459623) q[12];
ry(0.15937904943331802) q[13];
cx q[12],q[13];
ry(-1.3607483701784913) q[13];
ry(-1.86318193454174) q[14];
cx q[13],q[14];
ry(0.42266840198541517) q[13];
ry(0.2723818418952222) q[14];
cx q[13],q[14];
ry(-2.8661357040166653) q[14];
ry(-1.1910087902091127) q[15];
cx q[14],q[15];
ry(0.8824714348526738) q[14];
ry(1.978455871401805) q[15];
cx q[14],q[15];
ry(2.325587028951216) q[15];
ry(-2.726238987943551) q[16];
cx q[15],q[16];
ry(0.3123456162631629) q[15];
ry(0.1493840106444715) q[16];
cx q[15],q[16];
ry(1.2412917909630714) q[16];
ry(-0.6094780712450488) q[17];
cx q[16],q[17];
ry(-0.24663396463894038) q[16];
ry(2.8419039887881965) q[17];
cx q[16],q[17];
ry(1.0886974243704834) q[17];
ry(1.4614198322732923) q[18];
cx q[17],q[18];
ry(2.4813338982594386) q[17];
ry(0.4982457858657616) q[18];
cx q[17],q[18];
ry(-1.5306940058397562) q[18];
ry(1.6848429595912806) q[19];
cx q[18],q[19];
ry(-0.8888943251459299) q[18];
ry(-3.099611811130442) q[19];
cx q[18],q[19];
ry(1.066958469355141) q[0];
ry(-1.3105185832437884) q[1];
cx q[0],q[1];
ry(2.165937541452226) q[0];
ry(-2.3536374712426698) q[1];
cx q[0],q[1];
ry(1.3692201521410492) q[1];
ry(-1.9477650400054292) q[2];
cx q[1],q[2];
ry(-2.792450939149949) q[1];
ry(0.15880261341451413) q[2];
cx q[1],q[2];
ry(1.2128839037207637) q[2];
ry(-1.8995210375816782) q[3];
cx q[2],q[3];
ry(1.5571804511416012) q[2];
ry(1.2106159849227636) q[3];
cx q[2],q[3];
ry(1.5705446532109377) q[3];
ry(0.889666483156839) q[4];
cx q[3],q[4];
ry(1.4495009557105782) q[3];
ry(1.990103126248524) q[4];
cx q[3],q[4];
ry(1.5877990968215085) q[4];
ry(1.0859961160787295) q[5];
cx q[4],q[5];
ry(0.006290000678500718) q[4];
ry(2.5779359862181477) q[5];
cx q[4],q[5];
ry(1.7893016427192947) q[5];
ry(-1.8386555944830656) q[6];
cx q[5],q[6];
ry(-3.014477528167218) q[5];
ry(-1.609129374259141) q[6];
cx q[5],q[6];
ry(1.54985161493778) q[6];
ry(-1.7532369136715609) q[7];
cx q[6],q[7];
ry(0.662104111791705) q[6];
ry(-1.900433099306336) q[7];
cx q[6],q[7];
ry(-2.7827707140259266) q[7];
ry(1.125350801211595) q[8];
cx q[7],q[8];
ry(-0.5829573314459884) q[7];
ry(1.933390267626677) q[8];
cx q[7],q[8];
ry(2.6024563766047057) q[8];
ry(-2.936528631976204) q[9];
cx q[8],q[9];
ry(0.08575645683028685) q[8];
ry(-0.7962298632206295) q[9];
cx q[8],q[9];
ry(1.1206013207517067) q[9];
ry(1.4094026329423883) q[10];
cx q[9],q[10];
ry(-1.4782163311412342) q[9];
ry(-2.0102535149760277) q[10];
cx q[9],q[10];
ry(0.4412311684107804) q[10];
ry(1.211656094790311) q[11];
cx q[10],q[11];
ry(0.4517109321958174) q[10];
ry(1.3142520054134135) q[11];
cx q[10],q[11];
ry(-0.8830707203910269) q[11];
ry(0.7888694110687764) q[12];
cx q[11],q[12];
ry(0.1047960202938576) q[11];
ry(0.7771918685468828) q[12];
cx q[11],q[12];
ry(-2.3507263037893638) q[12];
ry(-1.843684590279019) q[13];
cx q[12],q[13];
ry(2.147630055470071) q[12];
ry(2.024259630848615) q[13];
cx q[12],q[13];
ry(3.1055914916827687) q[13];
ry(1.9629579835838757) q[14];
cx q[13],q[14];
ry(3.0587129360376917) q[13];
ry(1.9236439138259558) q[14];
cx q[13],q[14];
ry(1.597986340647373) q[14];
ry(-1.7935829320210246) q[15];
cx q[14],q[15];
ry(2.6291335694215) q[14];
ry(-0.787774172757679) q[15];
cx q[14],q[15];
ry(-0.6114998556166442) q[15];
ry(-1.2313712424431182) q[16];
cx q[15],q[16];
ry(-0.013335102307813778) q[15];
ry(2.3356792956328842) q[16];
cx q[15],q[16];
ry(-1.329658175058536) q[16];
ry(-1.6259237905973147) q[17];
cx q[16],q[17];
ry(-1.7909664634320648) q[16];
ry(1.242052705231357) q[17];
cx q[16],q[17];
ry(1.8764174271752192) q[17];
ry(-0.6825269819145489) q[18];
cx q[17],q[18];
ry(-0.5518191494194707) q[17];
ry(0.5394646315890332) q[18];
cx q[17],q[18];
ry(2.668501111402321) q[18];
ry(2.6500188479032047) q[19];
cx q[18],q[19];
ry(0.6184744604722514) q[18];
ry(-2.9871313037118967) q[19];
cx q[18],q[19];
ry(2.6609641719237582) q[0];
ry(1.5516174692892402) q[1];
cx q[0],q[1];
ry(0.9629594290296455) q[0];
ry(-0.44609310866632623) q[1];
cx q[0],q[1];
ry(1.8061603465243818) q[1];
ry(-1.285027801086207) q[2];
cx q[1],q[2];
ry(0.12019700358022518) q[1];
ry(-0.07996467045822424) q[2];
cx q[1],q[2];
ry(-1.8081331508630747) q[2];
ry(-1.5637032419335517) q[3];
cx q[2],q[3];
ry(-2.7766726319716777) q[2];
ry(2.822825455351856) q[3];
cx q[2],q[3];
ry(1.569129626690048) q[3];
ry(-2.092762615481761) q[4];
cx q[3],q[4];
ry(-0.10033259160827863) q[3];
ry(-2.9639992525679766) q[4];
cx q[3],q[4];
ry(2.093770860953681) q[4];
ry(-1.3001338634257982) q[5];
cx q[4],q[5];
ry(-1.5712550187953775) q[4];
ry(-0.49441600253635704) q[5];
cx q[4],q[5];
ry(1.576893868992042) q[5];
ry(-1.9127701038581164) q[6];
cx q[5],q[6];
ry(-1.5740405662791213) q[5];
ry(-2.9211179843558686) q[6];
cx q[5],q[6];
ry(1.5754001911171496) q[6];
ry(1.5195976678067307) q[7];
cx q[6],q[7];
ry(1.5649009527031201) q[6];
ry(0.18361650963496334) q[7];
cx q[6],q[7];
ry(1.5744113966788122) q[7];
ry(1.3548179676089949) q[8];
cx q[7],q[8];
ry(1.5709179677817227) q[7];
ry(-0.5275863719237472) q[8];
cx q[7],q[8];
ry(-1.5742785864889957) q[8];
ry(1.819645207164188) q[9];
cx q[8],q[9];
ry(1.5730174435100501) q[8];
ry(-0.2559396905358673) q[9];
cx q[8],q[9];
ry(1.571676270498224) q[9];
ry(-1.5478388103911558) q[10];
cx q[9],q[10];
ry(1.570820103814583) q[9];
ry(-0.179087126913811) q[10];
cx q[9],q[10];
ry(1.5710613896754255) q[10];
ry(1.3294022386550435) q[11];
cx q[10],q[11];
ry(1.5694142789281145) q[10];
ry(2.3559087690300378) q[11];
cx q[10],q[11];
ry(1.571661460092361) q[11];
ry(1.7630984126234348) q[12];
cx q[11],q[12];
ry(1.5695366347037059) q[11];
ry(0.22315612532149137) q[12];
cx q[11],q[12];
ry(-1.5673506203910759) q[12];
ry(-1.6121139080294467) q[13];
cx q[12],q[13];
ry(1.5699979301037184) q[12];
ry(-0.16750262375339314) q[13];
cx q[12],q[13];
ry(-1.5710414861583137) q[13];
ry(1.8320933876621397) q[14];
cx q[13],q[14];
ry(1.571879584136066) q[13];
ry(-1.3457350775054726) q[14];
cx q[13],q[14];
ry(-1.5618413632110126) q[14];
ry(-1.7493706992490314) q[15];
cx q[14],q[15];
ry(-1.571260537328839) q[14];
ry(-2.935892808243759) q[15];
cx q[14],q[15];
ry(-1.568920435193574) q[15];
ry(-1.5018393007234818) q[16];
cx q[15],q[16];
ry(-1.569496608437596) q[15];
ry(2.8765875505772422) q[16];
cx q[15],q[16];
ry(-1.5701654153031301) q[16];
ry(-2.040135245502478) q[17];
cx q[16],q[17];
ry(-1.569249528421737) q[16];
ry(1.663666705410976) q[17];
cx q[16],q[17];
ry(1.5714446474105663) q[17];
ry(-0.5848560946280339) q[18];
cx q[17],q[18];
ry(1.5720922091162746) q[17];
ry(-0.9978552651631559) q[18];
cx q[17],q[18];
ry(-1.5726769622470922) q[18];
ry(-2.2298301350422456) q[19];
cx q[18],q[19];
ry(1.5700779958079423) q[18];
ry(2.540252547992673) q[19];
cx q[18],q[19];
ry(-0.6300301526028604) q[0];
ry(-1.7876572939205948) q[1];
ry(-1.548409261569474) q[2];
ry(-1.5785104779732173) q[3];
ry(1.5610017449210598) q[4];
ry(1.5738286070321355) q[5];
ry(1.5607847917079596) q[6];
ry(1.5653578880055867) q[7];
ry(-1.569781051623207) q[8];
ry(1.570387975191142) q[9];
ry(1.571043914341546) q[10];
ry(1.5703434740789382) q[11];
ry(1.5663661878301847) q[12];
ry(1.5696527054426275) q[13];
ry(-1.5636816317610367) q[14];
ry(1.5731587093700237) q[15];
ry(1.5719333187740236) q[16];
ry(1.5726868398820482) q[17];
ry(-1.57082058972198) q[18];
ry(-1.5715247108385906) q[19];