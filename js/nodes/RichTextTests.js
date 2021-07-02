// Copyright 2021, University of Colorado Boulder

/**
 * RichText tests
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import StringProperty from '../../../axon/js/StringProperty.js';
import RichText from './RichText.js';

QUnit.module( 'RichText' );

QUnit.test( 'Mutually exclusive options', assert => {

  const textProperty = new StringProperty( 'um, hoss?' );
  window.assert && assert.throws( () => {
    return new RichText( {
      text: 'hi',
      textProperty: textProperty
    } );
  }, 'text and textProperty values do not match' );

} );

