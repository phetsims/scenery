// Copyright 2021-2023, University of Colorado Boulder

/**
 * RichText tests
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import StringProperty from '../../../axon/js/StringProperty.js';
import RichText from './RichText.js';

QUnit.module( 'RichText' );

QUnit.test( 'Mutually exclusive options', assert => {

  assert.ok( true, 'always true, even when assertions are not on.' );

  const stringProperty = new StringProperty( 'um, hoss?' );
  window.assert && assert.throws( () => {
    return new RichText( {

      // @ts-expect-error for testing
      string: 'hi',
      stringProperty: stringProperty
    } );
  }, 'text and stringProperty values do not match' );

} );

QUnit.test( 'DerivedProperty stringProperty', assert => {

  assert.ok( true, 'always true, even when assertions are not on.' );

  const string = 'oh boy, here we go';
  const stringProperty = new StringProperty( string );

  const extra = '!!';
  const aBitExtraForAStringProperty = new DerivedProperty( [ stringProperty ], value => value + extra );

  const text = new RichText( aBitExtraForAStringProperty );

  assert.ok( text.stringProperty.value === string + extra );
  stringProperty.value = string + extra;
  assert.ok( text.string === string + extra + extra );

  window.assert && assert.throws( () => {
    text.string = 'hi';
  }, 'cannot set a derivedProperty' );
} );