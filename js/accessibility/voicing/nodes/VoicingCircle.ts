// Copyright 2025, University of Colorado Boulder

/**
 * A Circle that mixes Voicing.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import { EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import { Circle, CircleOptions, ReadingBlockOptions, scenery, Voicing } from '../../../imports.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = ReadingBlockOptions & CircleOptions;
export type VoicingTextOptions = SelfOptions & ParentOptions;

class VoicingCircle extends Voicing( Circle ) {
  public constructor( radius?: number | CircleOptions, options?: CircleOptions ) {
    if ( typeof radius === 'object' || radius === undefined ) {
      super( options );
    }
    else {
      super( radius, options );
    }
  }
}

scenery.register( 'VoicingCircle', VoicingCircle );
export default VoicingCircle;