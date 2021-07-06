// Copyright 2021, University of Colorado Boulder

/**
 * Text tests
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import StringProperty from '../../../axon/js/StringProperty.js';
import Text from './Text.js';

QUnit.module( 'Text' );

QUnit.test( 'Mutually exclusive options', assert => {

  assert.ok( true, 'always true, even when assertions are not on.' );

  const textProperty = new StringProperty( 'oh boy, here we go.' );
  window.assert && assert.throws( () => {
    return new Text( {
      text: 'hi',
      textProperty: textProperty
    } );
  }, 'text and textProperty values do not match' );

} );

