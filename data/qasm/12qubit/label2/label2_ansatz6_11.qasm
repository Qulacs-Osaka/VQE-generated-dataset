OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.6377086391804485) q[0];
ry(-2.7675263088512616) q[1];
cx q[0],q[1];
ry(2.3833143864393334) q[0];
ry(0.4738032350187514) q[1];
cx q[0],q[1];
ry(0.26491483109235947) q[1];
ry(-2.167895152890011) q[2];
cx q[1],q[2];
ry(0.06539776822101065) q[1];
ry(-1.2377568593346657) q[2];
cx q[1],q[2];
ry(0.5877850120708362) q[2];
ry(2.678846368102359) q[3];
cx q[2],q[3];
ry(-0.044914525111256154) q[2];
ry(3.1376466817263733) q[3];
cx q[2],q[3];
ry(-2.9535661042015486) q[3];
ry(2.8834391900174303) q[4];
cx q[3],q[4];
ry(-0.10592467307664233) q[3];
ry(0.31847911347735725) q[4];
cx q[3],q[4];
ry(2.4659195735173323) q[4];
ry(-1.5529154602885686) q[5];
cx q[4],q[5];
ry(-0.9906201000719759) q[4];
ry(-0.0607502244272542) q[5];
cx q[4],q[5];
ry(1.1524092562674082) q[5];
ry(-0.3112431449339986) q[6];
cx q[5],q[6];
ry(-0.6736288047560498) q[5];
ry(-1.266880705335954) q[6];
cx q[5],q[6];
ry(-2.555008575170686) q[6];
ry(0.19027836110871624) q[7];
cx q[6],q[7];
ry(-3.1413659572571397) q[6];
ry(0.0003407449755659591) q[7];
cx q[6],q[7];
ry(-0.5630089492101595) q[7];
ry(0.4484083402982497) q[8];
cx q[7],q[8];
ry(1.4565562355782007) q[7];
ry(1.8137445644039802) q[8];
cx q[7],q[8];
ry(-1.1025305729996875) q[8];
ry(-1.5080272575394968) q[9];
cx q[8],q[9];
ry(1.3924192407406455) q[8];
ry(0.3301724473077323) q[9];
cx q[8],q[9];
ry(2.8653919908103767) q[9];
ry(-3.0685283309250937) q[10];
cx q[9],q[10];
ry(1.723832563006109) q[9];
ry(-1.7416045068954524) q[10];
cx q[9],q[10];
ry(-1.1936360338601462) q[10];
ry(-2.4013613352993595) q[11];
cx q[10],q[11];
ry(-1.4462603435613994) q[10];
ry(1.910380589513105) q[11];
cx q[10],q[11];
ry(0.34002783010529924) q[0];
ry(1.989616836174646) q[1];
cx q[0],q[1];
ry(-1.236773441732719) q[0];
ry(2.808754907051559) q[1];
cx q[0],q[1];
ry(-2.2185785796505764) q[1];
ry(0.08093544089141869) q[2];
cx q[1],q[2];
ry(2.6911455500252406) q[1];
ry(2.6814452902490893) q[2];
cx q[1],q[2];
ry(0.9230912396380937) q[2];
ry(2.3573159452425676) q[3];
cx q[2],q[3];
ry(0.3593388181452114) q[2];
ry(-0.8461688034391157) q[3];
cx q[2],q[3];
ry(-1.3689651549186062) q[3];
ry(2.559926790894132) q[4];
cx q[3],q[4];
ry(3.1380322547755415) q[3];
ry(0.003333409402942677) q[4];
cx q[3],q[4];
ry(0.16048121539074778) q[4];
ry(-0.6379324453243518) q[5];
cx q[4],q[5];
ry(0.03568039159630224) q[4];
ry(0.7031485792728257) q[5];
cx q[4],q[5];
ry(2.682417393048379) q[5];
ry(2.7830444821591174) q[6];
cx q[5],q[6];
ry(-1.7617456016880624) q[5];
ry(-3.090223763107653) q[6];
cx q[5],q[6];
ry(2.5932279926177095) q[6];
ry(-2.4407263009045197) q[7];
cx q[6],q[7];
ry(-0.43284189240618437) q[6];
ry(3.0747217861491833) q[7];
cx q[6],q[7];
ry(0.48728455913122204) q[7];
ry(-1.3206157183486855) q[8];
cx q[7],q[8];
ry(2.121599255813164) q[7];
ry(0.03447454977894031) q[8];
cx q[7],q[8];
ry(-1.3585598200941398) q[8];
ry(0.9564807298068748) q[9];
cx q[8],q[9];
ry(-2.5836703958734955) q[8];
ry(1.5382113894823783) q[9];
cx q[8],q[9];
ry(-1.0658875489012438) q[9];
ry(2.8846855444445234) q[10];
cx q[9],q[10];
ry(-1.6077672467228101) q[9];
ry(-3.13641981176922) q[10];
cx q[9],q[10];
ry(2.1709386329832325) q[10];
ry(-2.105915867135008) q[11];
cx q[10],q[11];
ry(-0.4199391994153154) q[10];
ry(0.5158944724238558) q[11];
cx q[10],q[11];
ry(-2.4264157386320853) q[0];
ry(-1.2569719460679072) q[1];
cx q[0],q[1];
ry(-0.7636771933997012) q[0];
ry(1.9400277954770955) q[1];
cx q[0],q[1];
ry(-2.6554570991521285) q[1];
ry(1.697008094448558) q[2];
cx q[1],q[2];
ry(1.5648448956568823) q[1];
ry(2.7493776687000566) q[2];
cx q[1],q[2];
ry(-1.3794162947853894) q[2];
ry(0.5090460211349359) q[3];
cx q[2],q[3];
ry(-1.0986871149182562) q[2];
ry(-0.2074416122122376) q[3];
cx q[2],q[3];
ry(-1.5029686694000304) q[3];
ry(2.8535860473004866) q[4];
cx q[3],q[4];
ry(-3.1253084937324007) q[3];
ry(-3.1413084355631784) q[4];
cx q[3],q[4];
ry(3.041213372288363) q[4];
ry(-0.3473636672944593) q[5];
cx q[4],q[5];
ry(-0.9964723887915126) q[4];
ry(2.2921391208336375) q[5];
cx q[4],q[5];
ry(1.440635971780815) q[5];
ry(-3.0055652216911497) q[6];
cx q[5],q[6];
ry(-0.0024501598926772563) q[5];
ry(3.120441751318721) q[6];
cx q[5],q[6];
ry(-0.6943545442292782) q[6];
ry(-0.7476318710013017) q[7];
cx q[6],q[7];
ry(2.8820030733781663) q[6];
ry(-0.6297355015333057) q[7];
cx q[6],q[7];
ry(3.08232012904075) q[7];
ry(-1.185398136340524) q[8];
cx q[7],q[8];
ry(-1.0997576911268698) q[7];
ry(0.02132251715391083) q[8];
cx q[7],q[8];
ry(1.5509784269994835) q[8];
ry(1.2133014208368011) q[9];
cx q[8],q[9];
ry(0.005413503175916823) q[8];
ry(1.777550163310464) q[9];
cx q[8],q[9];
ry(1.9185605685906066) q[9];
ry(-2.324816585581641) q[10];
cx q[9],q[10];
ry(0.42922512214196384) q[9];
ry(-1.1982652166945913) q[10];
cx q[9],q[10];
ry(2.3140770942894187) q[10];
ry(0.0015126449577920198) q[11];
cx q[10],q[11];
ry(-2.0495384346779106) q[10];
ry(0.7631656781821334) q[11];
cx q[10],q[11];
ry(-2.687588821092243) q[0];
ry(2.368064567243375) q[1];
cx q[0],q[1];
ry(-0.30771905426873863) q[0];
ry(-1.810603634796304) q[1];
cx q[0],q[1];
ry(1.03913534250297) q[1];
ry(-2.4471625783435678) q[2];
cx q[1],q[2];
ry(-1.7849020073067516) q[1];
ry(1.017950918636946) q[2];
cx q[1],q[2];
ry(1.5648897693395272) q[2];
ry(-2.4855923239590343) q[3];
cx q[2],q[3];
ry(-0.13132288281178983) q[2];
ry(0.6331357854233195) q[3];
cx q[2],q[3];
ry(1.4444231515917778) q[3];
ry(-2.4421463690150484) q[4];
cx q[3],q[4];
ry(-2.625339952911341) q[3];
ry(-0.9709159988897245) q[4];
cx q[3],q[4];
ry(-2.6016725647628838) q[4];
ry(-1.3002323349540936) q[5];
cx q[4],q[5];
ry(-1.147983637930207) q[4];
ry(0.17697015361796797) q[5];
cx q[4],q[5];
ry(1.6816787180031607) q[5];
ry(0.3350904970136437) q[6];
cx q[5],q[6];
ry(3.138703405862321) q[5];
ry(-3.1391752487777698) q[6];
cx q[5],q[6];
ry(1.0145202262632762) q[6];
ry(-2.046435024135196) q[7];
cx q[6],q[7];
ry(0.009684128878284959) q[6];
ry(0.9787538058946801) q[7];
cx q[6],q[7];
ry(0.9855335168754973) q[7];
ry(1.462280185659457) q[8];
cx q[7],q[8];
ry(-1.2994442200303589) q[7];
ry(-3.091291671378854) q[8];
cx q[7],q[8];
ry(-2.481403810631769) q[8];
ry(1.9710244065542177) q[9];
cx q[8],q[9];
ry(-0.0009005196478104808) q[8];
ry(3.1182977246494814) q[9];
cx q[8],q[9];
ry(-0.6453577661858156) q[9];
ry(-2.7183076804073343) q[10];
cx q[9],q[10];
ry(2.3846809081146914) q[9];
ry(-0.5193772373955643) q[10];
cx q[9],q[10];
ry(-0.6171459325779124) q[10];
ry(1.5365833908893167) q[11];
cx q[10],q[11];
ry(2.119045604919772) q[10];
ry(2.719622833744459) q[11];
cx q[10],q[11];
ry(-2.6699473363902815) q[0];
ry(-2.401368867386368) q[1];
cx q[0],q[1];
ry(-0.8447326991221589) q[0];
ry(-1.343869604917126) q[1];
cx q[0],q[1];
ry(-1.848808194378308) q[1];
ry(-2.659636545380302) q[2];
cx q[1],q[2];
ry(0.5274138275259741) q[1];
ry(0.4159351665977652) q[2];
cx q[1],q[2];
ry(3.0898221026805093) q[2];
ry(-1.2542489387622764) q[3];
cx q[2],q[3];
ry(-2.6814150093208267) q[2];
ry(2.1344112890795515) q[3];
cx q[2],q[3];
ry(0.980560051685302) q[3];
ry(3.0117973288323916) q[4];
cx q[3],q[4];
ry(3.126015118301002) q[3];
ry(0.32588270964134364) q[4];
cx q[3],q[4];
ry(0.5462676566312926) q[4];
ry(-2.272030303820573) q[5];
cx q[4],q[5];
ry(1.3345226829786463) q[4];
ry(-0.023539803174815965) q[5];
cx q[4],q[5];
ry(-0.41389980550299654) q[5];
ry(2.0096982633968468) q[6];
cx q[5],q[6];
ry(0.0005550654186166071) q[5];
ry(0.001773491040560035) q[6];
cx q[5],q[6];
ry(1.8354220082621353) q[6];
ry(2.56203270054259) q[7];
cx q[6],q[7];
ry(-2.0947488286175933) q[6];
ry(2.7117679869804956) q[7];
cx q[6],q[7];
ry(0.503194123391693) q[7];
ry(-1.3699282332774105) q[8];
cx q[7],q[8];
ry(-0.5845055734481281) q[7];
ry(-2.9196667903144147) q[8];
cx q[7],q[8];
ry(-0.693779549779976) q[8];
ry(-2.3726460004746057) q[9];
cx q[8],q[9];
ry(3.139866477906988) q[8];
ry(-3.1159896278870813) q[9];
cx q[8],q[9];
ry(-1.185344047793743) q[9];
ry(-2.192938191158561) q[10];
cx q[9],q[10];
ry(3.015703573122496) q[9];
ry(1.655152464880916) q[10];
cx q[9],q[10];
ry(-2.337507921178438) q[10];
ry(-0.2781008318396134) q[11];
cx q[10],q[11];
ry(-2.469525581890845) q[10];
ry(0.7726565280968868) q[11];
cx q[10],q[11];
ry(-1.927628522465205) q[0];
ry(-2.9761732098885467) q[1];
cx q[0],q[1];
ry(-1.8806125683272779) q[0];
ry(-1.3467012791678825) q[1];
cx q[0],q[1];
ry(0.2513281801182732) q[1];
ry(0.7965253553360984) q[2];
cx q[1],q[2];
ry(-0.032944820752597075) q[1];
ry(-1.8793727516898115) q[2];
cx q[1],q[2];
ry(-2.2658710687882926) q[2];
ry(1.9203570508973649) q[3];
cx q[2],q[3];
ry(-0.7024733103966074) q[2];
ry(-3.1034741559808383) q[3];
cx q[2],q[3];
ry(2.1236670880947193) q[3];
ry(0.8067942917160861) q[4];
cx q[3],q[4];
ry(0.6534785753816971) q[3];
ry(2.752679826655148) q[4];
cx q[3],q[4];
ry(-2.039284190015093) q[4];
ry(-2.3217303574102943) q[5];
cx q[4],q[5];
ry(0.08498804579647423) q[4];
ry(-0.09918422456478382) q[5];
cx q[4],q[5];
ry(3.015383335872568) q[5];
ry(-1.676103952809882) q[6];
cx q[5],q[6];
ry(0.0023798913281811804) q[5];
ry(-0.8761771690399938) q[6];
cx q[5],q[6];
ry(-0.1906450164837281) q[6];
ry(0.0465644892778867) q[7];
cx q[6],q[7];
ry(-3.026979012845078) q[6];
ry(0.025075158945362185) q[7];
cx q[6],q[7];
ry(-2.579817047203667) q[7];
ry(-1.5500838964625914) q[8];
cx q[7],q[8];
ry(2.964766371297838) q[7];
ry(-3.10280295215367) q[8];
cx q[7],q[8];
ry(-0.7894570860374923) q[8];
ry(-2.07079380691175) q[9];
cx q[8],q[9];
ry(3.027256062651026) q[8];
ry(0.022434106122034777) q[9];
cx q[8],q[9];
ry(2.229416658698251) q[9];
ry(3.1036497606364777) q[10];
cx q[9],q[10];
ry(0.14114898577142831) q[9];
ry(2.3457631697487002) q[10];
cx q[9],q[10];
ry(1.4261078070688793) q[10];
ry(-1.8760231780432877) q[11];
cx q[10],q[11];
ry(-0.9818932280935032) q[10];
ry(1.760051263551242) q[11];
cx q[10],q[11];
ry(-3.1273639608205017) q[0];
ry(-0.6590830470635147) q[1];
cx q[0],q[1];
ry(2.1127943651403696) q[0];
ry(0.11279551965770356) q[1];
cx q[0],q[1];
ry(2.340817229528572) q[1];
ry(0.880634477151101) q[2];
cx q[1],q[2];
ry(0.23502868946067323) q[1];
ry(2.0904310197215406) q[2];
cx q[1],q[2];
ry(2.861511925471593) q[2];
ry(-2.2320004277249623) q[3];
cx q[2],q[3];
ry(1.5598203465574823) q[2];
ry(-2.139323636377707) q[3];
cx q[2],q[3];
ry(0.22603003793245335) q[3];
ry(1.419404645510766) q[4];
cx q[3],q[4];
ry(1.2460814211247975) q[3];
ry(-0.8011716527557218) q[4];
cx q[3],q[4];
ry(-0.06821884039744999) q[4];
ry(1.5797216098365892) q[5];
cx q[4],q[5];
ry(-2.311460192853707) q[4];
ry(-0.0031632867665001285) q[5];
cx q[4],q[5];
ry(-3.0535487256121865) q[5];
ry(-1.3889281523238768) q[6];
cx q[5],q[6];
ry(3.109474330007437) q[5];
ry(0.6535483606011824) q[6];
cx q[5],q[6];
ry(0.2903071062543585) q[6];
ry(-0.8907399170784185) q[7];
cx q[6],q[7];
ry(-3.0046218607791713) q[6];
ry(-0.006986119844934541) q[7];
cx q[6],q[7];
ry(1.6579421505432097) q[7];
ry(2.78910380182954) q[8];
cx q[7],q[8];
ry(-0.10020939690787678) q[7];
ry(-1.795478333433279) q[8];
cx q[7],q[8];
ry(1.524935418893517) q[8];
ry(2.2267642893077553) q[9];
cx q[8],q[9];
ry(-1.8405028042255114) q[8];
ry(-0.22648359445363395) q[9];
cx q[8],q[9];
ry(-0.2409157517265843) q[9];
ry(-2.0451864513692213) q[10];
cx q[9],q[10];
ry(-1.9665350533669264) q[9];
ry(0.030930457342508433) q[10];
cx q[9],q[10];
ry(1.551222964591654) q[10];
ry(2.1602233909211925) q[11];
cx q[10],q[11];
ry(1.9994627817046722) q[10];
ry(0.11879839656596437) q[11];
cx q[10],q[11];
ry(-0.8487203404030783) q[0];
ry(-0.10923590975389351) q[1];
cx q[0],q[1];
ry(-2.9640256163985557) q[0];
ry(0.9197911215228326) q[1];
cx q[0],q[1];
ry(0.18319084724605536) q[1];
ry(-2.4640704221202823) q[2];
cx q[1],q[2];
ry(2.886123457129389) q[1];
ry(2.6791755869069274) q[2];
cx q[1],q[2];
ry(0.7864727462569501) q[2];
ry(-1.0475388120809086) q[3];
cx q[2],q[3];
ry(-3.050925358431417) q[2];
ry(-0.9936813453983131) q[3];
cx q[2],q[3];
ry(1.8281489728308653) q[3];
ry(-0.8517392675339569) q[4];
cx q[3],q[4];
ry(-0.9998883891589889) q[3];
ry(0.3289895342101158) q[4];
cx q[3],q[4];
ry(-2.449734555109297) q[4];
ry(0.5502831096382557) q[5];
cx q[4],q[5];
ry(-2.3283148500935873) q[4];
ry(3.106007322430539) q[5];
cx q[4],q[5];
ry(-0.09628809119405267) q[5];
ry(2.608143701769328) q[6];
cx q[5],q[6];
ry(3.0677421731733636) q[5];
ry(-0.39787004157960726) q[6];
cx q[5],q[6];
ry(-0.11162427000697672) q[6];
ry(-1.4508802370293914) q[7];
cx q[6],q[7];
ry(0.814594166945986) q[6];
ry(-0.0009852866790458555) q[7];
cx q[6],q[7];
ry(0.8610955728355552) q[7];
ry(0.9723937674389491) q[8];
cx q[7],q[8];
ry(-2.964709245447925) q[7];
ry(-3.0815004380509894) q[8];
cx q[7],q[8];
ry(2.76558477737919) q[8];
ry(2.8288987628381177) q[9];
cx q[8],q[9];
ry(2.692797310523827) q[8];
ry(-3.1266693083552703) q[9];
cx q[8],q[9];
ry(-0.5120023508855249) q[9];
ry(0.722466508162404) q[10];
cx q[9],q[10];
ry(3.119377580982245) q[9];
ry(0.05904649022381215) q[10];
cx q[9],q[10];
ry(2.4791374634276386) q[10];
ry(2.0215567676127497) q[11];
cx q[10],q[11];
ry(-0.02159671513851258) q[10];
ry(-0.662975984831669) q[11];
cx q[10],q[11];
ry(0.4416011708695138) q[0];
ry(0.8624420573746928) q[1];
cx q[0],q[1];
ry(0.486673471138833) q[0];
ry(-3.137736516240024) q[1];
cx q[0],q[1];
ry(0.25840228535500126) q[1];
ry(-0.45440643379376405) q[2];
cx q[1],q[2];
ry(-0.3785774792827972) q[1];
ry(-0.3776029755450905) q[2];
cx q[1],q[2];
ry(-2.7775529168243986) q[2];
ry(1.9504279124781683) q[3];
cx q[2],q[3];
ry(-0.44578453233363113) q[2];
ry(1.7468686396331305) q[3];
cx q[2],q[3];
ry(2.752131962575797) q[3];
ry(-3.124753492077029) q[4];
cx q[3],q[4];
ry(3.0215402768322424) q[3];
ry(-1.666452490139341) q[4];
cx q[3],q[4];
ry(-1.253674804197176) q[4];
ry(-0.8339964275087668) q[5];
cx q[4],q[5];
ry(3.031714613288244) q[4];
ry(3.1361680872519044) q[5];
cx q[4],q[5];
ry(-1.4752944547437896) q[5];
ry(-2.4405915811759384) q[6];
cx q[5],q[6];
ry(2.733550951030971) q[5];
ry(0.08934682651444437) q[6];
cx q[5],q[6];
ry(2.9255588006731634) q[6];
ry(2.772084290730475) q[7];
cx q[6],q[7];
ry(3.1264229275853994) q[6];
ry(0.00271064722137343) q[7];
cx q[6],q[7];
ry(-2.879278460966331) q[7];
ry(-1.6115246513351145) q[8];
cx q[7],q[8];
ry(0.7180019506045733) q[7];
ry(2.026585456199583) q[8];
cx q[7],q[8];
ry(-2.982838797013614) q[8];
ry(-1.7278195268777106) q[9];
cx q[8],q[9];
ry(-1.7994217705028177) q[8];
ry(1.1176388180429133) q[9];
cx q[8],q[9];
ry(2.3272352370189546) q[9];
ry(-2.374643343963171) q[10];
cx q[9],q[10];
ry(-0.014402968037289021) q[9];
ry(-0.00528426406294713) q[10];
cx q[9],q[10];
ry(1.2345088554584285) q[10];
ry(0.2786095803304013) q[11];
cx q[10],q[11];
ry(2.4631107159819923) q[10];
ry(0.7424022941666107) q[11];
cx q[10],q[11];
ry(0.9346344740414709) q[0];
ry(-0.1305558310626278) q[1];
cx q[0],q[1];
ry(-0.13827524082476295) q[0];
ry(-2.7567123279422177) q[1];
cx q[0],q[1];
ry(1.7773670431318225) q[1];
ry(-1.4029223127623789) q[2];
cx q[1],q[2];
ry(-2.465628041862481) q[1];
ry(-1.6524564553679293) q[2];
cx q[1],q[2];
ry(1.255355234143799) q[2];
ry(1.757797270561043) q[3];
cx q[2],q[3];
ry(3.1148551757782985) q[2];
ry(-0.0068867232274456925) q[3];
cx q[2],q[3];
ry(-0.822465946596683) q[3];
ry(-0.2717359766807066) q[4];
cx q[3],q[4];
ry(-0.8974232277252315) q[3];
ry(-0.799609994140118) q[4];
cx q[3],q[4];
ry(-0.17877216703569473) q[4];
ry(-1.694433170095605) q[5];
cx q[4],q[5];
ry(-0.026471537556870105) q[4];
ry(3.1380981653162037) q[5];
cx q[4],q[5];
ry(-1.2105846639303381) q[5];
ry(-1.3001930712992569) q[6];
cx q[5],q[6];
ry(-3.113909959487969) q[5];
ry(1.3724126753626065) q[6];
cx q[5],q[6];
ry(-0.38419452855523856) q[6];
ry(-1.9267287989992472) q[7];
cx q[6],q[7];
ry(-0.17354728200059988) q[6];
ry(0.379679437436633) q[7];
cx q[6],q[7];
ry(-1.4779436454575403) q[7];
ry(-2.1016453110791486) q[8];
cx q[7],q[8];
ry(2.2122987940424625) q[7];
ry(-0.010291328529515198) q[8];
cx q[7],q[8];
ry(2.3332605383586085) q[8];
ry(-1.9447156411875413) q[9];
cx q[8],q[9];
ry(-2.5878004856274486) q[8];
ry(-2.265550724099686) q[9];
cx q[8],q[9];
ry(-3.003242768615503) q[9];
ry(-1.7877418190233227) q[10];
cx q[9],q[10];
ry(2.241260928401089) q[9];
ry(1.5677013327927956) q[10];
cx q[9],q[10];
ry(-0.7642424398651189) q[10];
ry(-2.425676132287098) q[11];
cx q[10],q[11];
ry(-0.7701316519978629) q[10];
ry(2.5431982636094324) q[11];
cx q[10],q[11];
ry(0.8523740904966516) q[0];
ry(2.4517019916360194) q[1];
cx q[0],q[1];
ry(-0.6894450862212125) q[0];
ry(-2.051183229505564) q[1];
cx q[0],q[1];
ry(1.257236534984814) q[1];
ry(1.3051139161365057) q[2];
cx q[1],q[2];
ry(-0.5775816912376097) q[1];
ry(0.9743945916902099) q[2];
cx q[1],q[2];
ry(-0.1900142778915992) q[2];
ry(-1.2314605024798384) q[3];
cx q[2],q[3];
ry(0.2872372617328667) q[2];
ry(-3.0586389167812555) q[3];
cx q[2],q[3];
ry(0.6020846543736251) q[3];
ry(-3.0575660520662367) q[4];
cx q[3],q[4];
ry(-2.553623823791251) q[3];
ry(2.4513717906086905) q[4];
cx q[3],q[4];
ry(1.7941570369848518) q[4];
ry(-0.6392975656193158) q[5];
cx q[4],q[5];
ry(3.1076602412265464) q[4];
ry(0.030144282461439853) q[5];
cx q[4],q[5];
ry(-2.3933354936762568) q[5];
ry(-1.0157769262339271) q[6];
cx q[5],q[6];
ry(-3.077200298085361) q[5];
ry(0.33770845973301805) q[6];
cx q[5],q[6];
ry(1.055649035514298) q[6];
ry(2.640346400804789) q[7];
cx q[6],q[7];
ry(1.568462218335351) q[6];
ry(-0.9285809187179455) q[7];
cx q[6],q[7];
ry(1.1626498810208368) q[7];
ry(2.1498297707602907) q[8];
cx q[7],q[8];
ry(-0.03661796988254136) q[7];
ry(3.1338656187427123) q[8];
cx q[7],q[8];
ry(-2.389327717308147) q[8];
ry(-1.6122753955882767) q[9];
cx q[8],q[9];
ry(0.012714140695005321) q[8];
ry(-3.129037287951013) q[9];
cx q[8],q[9];
ry(-1.4008922519657914) q[9];
ry(1.4088051404428135) q[10];
cx q[9],q[10];
ry(-0.7518465702915389) q[9];
ry(1.1119975189114166) q[10];
cx q[9],q[10];
ry(-1.3182844634078057) q[10];
ry(1.7798740649306621) q[11];
cx q[10],q[11];
ry(-0.9986664289494229) q[10];
ry(-2.239932360627739) q[11];
cx q[10],q[11];
ry(-0.4611970653985722) q[0];
ry(1.6263668909619788) q[1];
cx q[0],q[1];
ry(1.4633392610797678) q[0];
ry(2.9579613879150077) q[1];
cx q[0],q[1];
ry(-2.0676534990136224) q[1];
ry(0.1340957370979874) q[2];
cx q[1],q[2];
ry(-0.6392152546752493) q[1];
ry(-2.922666179087954) q[2];
cx q[1],q[2];
ry(-0.9762827150400847) q[2];
ry(0.4926829383536271) q[3];
cx q[2],q[3];
ry(-3.090821480287021) q[2];
ry(-0.012797531894583436) q[3];
cx q[2],q[3];
ry(-1.840515037223451) q[3];
ry(0.8043126385009781) q[4];
cx q[3],q[4];
ry(0.06399745406119714) q[3];
ry(0.5378197919618852) q[4];
cx q[3],q[4];
ry(2.62894268899922) q[4];
ry(1.1221455243365952) q[5];
cx q[4],q[5];
ry(-0.6846172940028596) q[4];
ry(-3.135240315746066) q[5];
cx q[4],q[5];
ry(2.4526425767445508) q[5];
ry(-1.4609833307716564) q[6];
cx q[5],q[6];
ry(3.141176246369359) q[5];
ry(0.004599605857299097) q[6];
cx q[5],q[6];
ry(2.1360700603033465) q[6];
ry(-2.094356645142203) q[7];
cx q[6],q[7];
ry(1.4968563048393582) q[6];
ry(2.391527545501178) q[7];
cx q[6],q[7];
ry(3.123997888991707) q[7];
ry(-2.312819272515014) q[8];
cx q[7],q[8];
ry(1.6703276464003993) q[7];
ry(-0.6020719287613403) q[8];
cx q[7],q[8];
ry(-1.1861982986945128) q[8];
ry(1.4942983016107831) q[9];
cx q[8],q[9];
ry(-1.9477021617472023) q[8];
ry(3.13841810269672) q[9];
cx q[8],q[9];
ry(-1.4851534194304534) q[9];
ry(2.280101177720799) q[10];
cx q[9],q[10];
ry(-1.2195443134151844) q[9];
ry(-2.409690200460025) q[10];
cx q[9],q[10];
ry(-0.9995469094819702) q[10];
ry(-0.38337643751231987) q[11];
cx q[10],q[11];
ry(-2.741654579182033) q[10];
ry(0.28503115256872236) q[11];
cx q[10],q[11];
ry(1.0263673333797332) q[0];
ry(0.02513901200844596) q[1];
cx q[0],q[1];
ry(3.0667468340633395) q[0];
ry(-2.736959956660098) q[1];
cx q[0],q[1];
ry(-1.0159297373519256) q[1];
ry(0.904289692795861) q[2];
cx q[1],q[2];
ry(-2.5156069627570052) q[1];
ry(1.6903744177353524) q[2];
cx q[1],q[2];
ry(0.8846419901116948) q[2];
ry(-1.9371663762827636) q[3];
cx q[2],q[3];
ry(2.2100278203145987) q[2];
ry(3.099455879682158) q[3];
cx q[2],q[3];
ry(2.6863955468820953) q[3];
ry(-1.9776742532622453) q[4];
cx q[3],q[4];
ry(-0.013831028202837992) q[3];
ry(-2.0972632688739212) q[4];
cx q[3],q[4];
ry(1.3377844432810881) q[4];
ry(-2.4566502312025063) q[5];
cx q[4],q[5];
ry(-0.6376673051984829) q[4];
ry(0.0005784730695985729) q[5];
cx q[4],q[5];
ry(1.2864854293996144) q[5];
ry(2.5051174150513416) q[6];
cx q[5],q[6];
ry(-2.625971698714319) q[5];
ry(-2.9915834803602084) q[6];
cx q[5],q[6];
ry(-1.9582401880789524) q[6];
ry(1.378060874926227) q[7];
cx q[6],q[7];
ry(1.7278621391792992) q[6];
ry(0.013297710532715001) q[7];
cx q[6],q[7];
ry(-1.729768186999786) q[7];
ry(-2.3729403512789795) q[8];
cx q[7],q[8];
ry(0.00030807699763989307) q[7];
ry(2.573804448559661) q[8];
cx q[7],q[8];
ry(-2.9422067367488403) q[8];
ry(-1.5519474634138177) q[9];
cx q[8],q[9];
ry(1.3896977784561164) q[8];
ry(-2.2629600994249124) q[9];
cx q[8],q[9];
ry(-2.258647411563035) q[9];
ry(-2.8522357963795755) q[10];
cx q[9],q[10];
ry(-2.7912639542836493) q[9];
ry(3.1342553129510686) q[10];
cx q[9],q[10];
ry(2.0807159163583124) q[10];
ry(1.8163797741094783) q[11];
cx q[10],q[11];
ry(3.0354743003772486) q[10];
ry(0.34548970102040055) q[11];
cx q[10],q[11];
ry(1.0231515585399042) q[0];
ry(0.9680724961057273) q[1];
cx q[0],q[1];
ry(1.2624544756234082) q[0];
ry(0.5192178844317374) q[1];
cx q[0],q[1];
ry(-1.1921425364874736) q[1];
ry(2.1572837304053607) q[2];
cx q[1],q[2];
ry(3.104688162368421) q[1];
ry(-0.9397148930766235) q[2];
cx q[1],q[2];
ry(2.153140299212219) q[2];
ry(1.9668849262898123) q[3];
cx q[2],q[3];
ry(-2.0657546031770897) q[2];
ry(-0.5424980870253987) q[3];
cx q[2],q[3];
ry(2.7450353797210574) q[3];
ry(-1.8156132831716176) q[4];
cx q[3],q[4];
ry(2.7807312134344895) q[3];
ry(1.9085199369108312) q[4];
cx q[3],q[4];
ry(-2.04941388768731) q[4];
ry(0.7563538124187342) q[5];
cx q[4],q[5];
ry(-0.9114955991256632) q[4];
ry(3.1009010603011884) q[5];
cx q[4],q[5];
ry(-1.38254119610946) q[5];
ry(1.3771875548196142) q[6];
cx q[5],q[6];
ry(-0.02540022619880611) q[5];
ry(-1.7021772246247826) q[6];
cx q[5],q[6];
ry(1.5914362518384761) q[6];
ry(-1.9521230019301696) q[7];
cx q[6],q[7];
ry(-2.366359570241412) q[6];
ry(0.7760568295463486) q[7];
cx q[6],q[7];
ry(-0.5769432665428592) q[7];
ry(2.495175690063447) q[8];
cx q[7],q[8];
ry(3.110438232400773) q[7];
ry(3.0899863729182204) q[8];
cx q[7],q[8];
ry(-2.47071730362483) q[8];
ry(0.6445845062881834) q[9];
cx q[8],q[9];
ry(0.0716786051163929) q[8];
ry(3.1051209317394375) q[9];
cx q[8],q[9];
ry(0.37230239227993067) q[9];
ry(2.03839083933483) q[10];
cx q[9],q[10];
ry(1.52676631827527) q[9];
ry(3.109425620652353) q[10];
cx q[9],q[10];
ry(-0.9216920344510754) q[10];
ry(0.22944302595672372) q[11];
cx q[10],q[11];
ry(0.730384705761499) q[10];
ry(2.6894330644134556) q[11];
cx q[10],q[11];
ry(-2.939798429723986) q[0];
ry(1.7478049296494993) q[1];
ry(0.47229388560826924) q[2];
ry(-2.3752420413436357) q[3];
ry(2.2840342316206828) q[4];
ry(-1.7676303564600178) q[5];
ry(-0.11221707694999412) q[6];
ry(0.43156920547130007) q[7];
ry(-1.2934962575230262) q[8];
ry(-1.990249599100185) q[9];
ry(2.0350545189652127) q[10];
ry(-0.134441259730527) q[11];