// Copyright 2016, University of Colorado Boulder

$( window ).ready( function() {
  'use strict';
  
  var element;

  function newElement() {
    element = document.createElement( 'div' );
    $( element ).css( 'float', 'left' );
    document.body.appendChild( element );
  }

  function string( str ) {
    element.appendChild( document.createTextNode( str ) );
    element.appendChild( document.createElement( 'div' ) );
  }

  // unencoded
  newElement();
  string( 'Unencoded' );
  string( '---' );
  string( 'بيروكسيد الهيدروجين' ); // ar
  string( 'Peròxid d\'hidrogen' ); // ca
  string( 'Υπεροξείδιο υδρογόνου' ); // el
  string( 'peróxido de hidrógeno' ); // es
  string( 'هیدروژن پراکسید' ); // fa
  string( 'Peroxyde d\'hydrogène' ); // fr
  string( 'Peróxido de Hidróxeno' ); // gl
  string( 'Hidrogén-peroxid' ); // hu
  string( 'מי חמצן' ); // iw
  string( '過酸化水素' ); // ja
  string( 'អ៊ីដ្រូសែនពែអុកស៊ីត' ); // km
  string( '과산화 수소' ); // ko
  string( 'Водород-пероксид' ); // mk
  string( 'हायड्रोजन पेरॉक्साईड' ); // mr
  string( 'Peróxido de Hidrogênio' ); // pt_BR
  string( 'Пероксид водорода' ); // ru
  string( 'Водоник-пероксид' ); // sr
  string( 'Väteperoxid' ); // sv
  string( 'ไฮโดรเจนเปอร์ออกไซด์' ); // th
  string( 'Wodorodyň öteturşusy' ); // tk
  string( 'Kéo phân tử của bạn đến đây' ); // vi (from other string)
  string( '过氧化氢' ); // zh_CN
  string( '過氧化氫' ); // zh_TW
  string( '∫' );
  string( 'ﷺ' );
  string( '§' );
  string( 'Á' );
  string( 'ÿ' );
  string( 'Ω' );
  string( 'आ' );
  string( '私' );
  string( '達' );
  string( 'Å̳̥͓͚͒͞͞' );
  string( '0҉' );
  string( '█' );

  // encoded
  newElement();
  string( 'Encoded' );
  string( '---' );
  string( '\u0628\u064A\u0631\u0648\u0643\u0633\u064A\u062F \u0627\u0644\u0647\u064A\u062F\u0631\u0648\u062C\u064A\u0646' ); // ar
  string( 'Per\u00F2xid d\'hidrogen' ); // ca
  string( '\u03A5\u03C0\u03B5\u03C1\u03BF\u03BE\u03B5\u03AF\u03B4\u03B9\u03BF \u03C5\u03B4\u03C1\u03BF\u03B3\u03CC\u03BD\u03BF\u03C5' ); // el
  string( 'per\u00F3xido de hidr\u00F3geno' ); // es
  string( '\u0647\u06CC\u062F\u0631\u0648\u0698\u0646 \u067E\u0631\u0627\u06A9\u0633\u06CC\u062F' ); // fa
  string( 'Peroxyde d\'hydrog\u00E8ne' ); // fr
  string( 'Per\u00F3xido de Hidr\u00F3xeno' ); // gl
  string( 'Hidrog\u00E9n-peroxid' ); // hu
  string( '\u05DE\u05D9 \u05D7\u05DE\u05E6\u05DF' ); // iw
  string( '\u904E\u9178\u5316\u6C34\u7D20' ); // ja
  string( '\u17A2\u17CA\u17B8\u178A\u17D2\u179A\u17BC\u179F\u17C2\u1793\u1796\u17C2\u17A2\u17BB\u1780\u179F\u17CA\u17B8\u178F' ); // km
  string( '\uACFC\uC0B0\uD654 \uC218\uC18C' ); // ko
  string( '\u0412\u043E\u0434\u043E\u0440\u043E\u0434-\u043F\u0435\u0440\u043E\u043A\u0441\u0438\u0434' ); // mk
  string( '\u0939\u093E\u092F\u0921\u094D\u0930\u094B\u091C\u0928 \u092A\u0947\u0930\u0949\u0915\u094D\u0938\u093E\u0908\u0921' ); // mr
  string( 'Per\u00F3xido de Hidrog\u00EAnio' ); // pt_BR
  string( '\u041F\u0435\u0440\u043E\u043A\u0441\u0438\u0434 \u0432\u043E\u0434\u043E\u0440\u043E\u0434\u0430' ); // ru
  string( '\u0412\u043E\u0434\u043E\u043D\u0438\u043A-\u043F\u0435\u0440\u043E\u043A\u0441\u0438\u0434' ); // sr
  string( 'V\u00E4teperoxid' ); // sv
  string( '\u0E44\u0E2E\u0E42\u0E14\u0E23\u0E40\u0E08\u0E19\u0E40\u0E1B\u0E2D\u0E23\u0E4C\u0E2D\u0E2D\u0E01\u0E44\u0E0B\u0E14\u0E4C' ); // th
  string( 'Wodorody\u0148 \u00F6tetur\u015Fusy' ); // tk
  string( 'K\u00E9o ph\u00E2n t\u1EED c\u1EE7a b\u1EA1n \u0111\u1EBFn \u0111\u00E2y' ); // vi (from other string)
  string( '\u8FC7\u6C27\u5316\u6C22' ); // zh_CN
  string( '\u904E\u6C27\u5316\u6C2B' ); // zh_TW
  string( '\u222b' );
  string( '\ufdfa' );
  string( '\u00a7' );
  string( '\u00C1' );
  string( '\u00FF' );
  string( '\u03A9' );
  string( '\u0906' );
  string( '\u79C1' );
  string( '\u9054' );
  string( 'A\u030a\u0352\u0333\u0325\u0353\u035a\u035e\u035e' );
  string( '0\u0489' );
  string( '\u2588' );

} );
