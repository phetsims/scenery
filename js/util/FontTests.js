// Copyright 2018-2020, University of Colorado Boulder

/**
 * Font tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Font from './Font.js';

QUnit.module( 'Font' );

QUnit.test( 'Font.fromCSS', assert => {
  const font1 = Font.fromCSS( 'italic 1.2em "Fira Sans", sans-serif' );
  assert.equal( font1.style, 'italic' );
  assert.equal( font1.size, '1.2em' );
  assert.equal( font1.family, '"Fira Sans", sans-serif' );

  const font2 = Font.fromCSS( 'italic small-caps bold 16px/2 cursive' );
  assert.equal( font2.style, 'italic' );
  assert.equal( font2.variant, 'small-caps' );
  assert.equal( font2.weight, 'bold' );
  assert.equal( font2.size, '16px' );
  assert.equal( font2.lineHeight, '2' );
  assert.equal( font2.family, 'cursive' );
} );