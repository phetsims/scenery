// Copyright 2025, University of Colorado Boulder

/**
 * A superclass Node that is composed with Voicing, for where inheritance is a preferred pattern.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import { EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import Node, { NodeOptions } from '../../../nodes/Node.js';
import scenery from '../../../scenery.js';
import Voicing, { VoicingOptions } from '../Voicing.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = VoicingOptions & NodeOptions;
export type VoicingNodeOptions = SelfOptions & ParentOptions;

class VoicingNode extends Voicing( Node ) {
  public constructor( providedOptions?: VoicingNodeOptions ) {
    super( providedOptions );
  }
}

scenery.register( 'VoicingNode', VoicingNode );
export default VoicingNode;