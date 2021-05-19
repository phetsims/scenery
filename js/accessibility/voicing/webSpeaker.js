// Copyright 2020, University of Colorado Boulder

/**
 * Uses the Web Speech API to produce speech from the browser. This is a prototype, DO NOT USE IN PRODUCTION CODE.
 * There is no speech output until the webSpeaker has been initialized. Supported voices will depend on platform.
 * For each voice, you can customize the rate and pitch. Only one webSpeaker should be active at a time and so this
 * type is a singleton.
 *
 * @author Jesse Greenberg
 */

import BooleanProperty from '../../../../axon/js/BooleanProperty.js';
import DerivedProperty from '../../../../axon/js/DerivedProperty.js';
import Emitter from '../../../../axon/js/Emitter.js';
import NumberProperty from '../../../../axon/js/NumberProperty.js';
import Property from '../../../../axon/js/Property.js';
import stepTimer from '../../../../axon/js/stepTimer.js';
import Range from '../../../../dot/js/Range.js';
import platform from '../../../../phet-core/js/platform.js';
import stripEmbeddingMarks from '../../../../phet-core/js/stripEmbeddingMarks.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import VoicingUtterance from '../../../../utterance-queue/js/VoicingUtterance.js';
import scenery from '../../scenery.js';
import globalKeyStateTracker from '../globalKeyStateTracker.js';
import KeyboardUtils from '../KeyboardUtils.js';

class WebSpeaker {
  constructor() {

    // @public {null|SpeechSynthesisVoice}
    this.voiceProperty = new Property( null );

    // @public {NumberProperty} - controls the speaking rate of Web Speech
    this.voiceRateProperty = new NumberProperty( 1.0, { range: new Range( 0.75, 2 ) } );

    // {NumberProperty} - controls the pitch of the synth
    this.voicePitchProperty = new NumberProperty( 1.0, { range: new Range( 0.5, 2 ) } );

    // @public {Emitter} - emits events when the speaker starts/stops speaking, with the Utterance that is
    // either starting or stopping
    this.startSpeakingEmitter = new Emitter( { parameters: [ { valueType: Utterance } ] } );
    this.endSpeakingEmitter = new Emitter( { parameters: [ { valueType: Utterance } ] } );

    // @public - whether or not the synth is speaking - perhaps this should
    // replace the emitters above?
    this.speakingProperty = new BooleanProperty( false );

    // {SpeechSynthesis|null} - synth from Web Speech API that drives speech
    this.synth = null;

    // @public {SpeechSynthesisVoice[]} - possible voices for Web Speech synthesis
    this.voices = [];

    // @public {boolean} - is the WebSpeaker initialized for use? This is prototypal so it isn't always initialized
    this.initialized = false;

    // whether or ot the webSpeaker is enabled - if false, there will be no speech
    this.enabledProperty = new BooleanProperty( false );

    // @private {DerivedProperty} - Controls whether or not speech is allowed with synthesis.
    // This controlling Property can be set with setCanSpeakProperty.
    this._canSpeakProperty = new DerivedProperty( [ this.enabledProperty ], value => value );

    // @private {function} - bound so we can link and unlink to canSpeakProperty if a new canSpeakProperty is set
    this.boundHandleCanSpeakChange = this.handleCanSpeakChange.bind( this );
    this._canSpeakProperty.link( this.boundHandleCanSpeakChange );

    // @private {Utterance} - A reference to the last utterance spoken, so we can determine
    // cancelling behavior when it is time to speak the next utterance. See VoicingUtterance options.
    this.previousUtterance = null;

    // @public {boolean} - a more interal way to disable speaking - the enabledProperty
    // can be set by the user and is publicly observable for other things - but if
    // you need to temporarily shut down speaking without changing that observable
    // you can set onHold to true to prevent all speaking. Useful in cases like
    // the ResetAllButton where you want to describe the reset without
    // any of the other changing Properties in that interaction
    this.onHold = false;

    // fixes a bug on Safari where the `start` and `end` Utterances don't fire! The
    // issue is (apparently) that Safari internally clears the reference to the
    // Utterance on speak which prevents it from firing these events at the right
    // time - fix borrowed from
    // https://stackoverflow.com/questions/23483990/speechsynthesis-api-onend-callback-not-working
    // Unfortunately, this also introduces a memory leak, we should be smarter about
    // clearing this, though it is a bit tricky since we don't have a way to know
    // when we are done with an utterance - see #215
    this.utterances = [];
  }

  get enabled() {
    return this.enabledProperty.get();
  }

  /**
   * Indicate that the webSpeaker is ready for use, and attempt to populate voices (if they are ready yet). Adds
   * listeners that control speech.
   * @public
   */
  initialize() {
    assert && assert( this.initialized === false, 'can only be initialized once' );
    this.initialized = true;

    this.synth = window.speechSynthesis;
    assert && assert( this.synth, 'SpeechSynthesis not supported on your platform.' );

    // On chrome, synth.getVoices() returns an empty array until the onvoiceschanged event, so we have to
    // wait to populate
    this.synth.onvoiceschanged = () => {
      this.populateVoices();

      // remove the listener after they have been populated once from this event
      this.synth.onvoiceschanged = null;
    };

    // otherwise, try to populate voices immediately
    this.populateVoices();

    // The control key will stop the synth from speaking if there is an active utterance. This key was decided because
    // most major screen readers will stop speech when this key is pressed
    globalKeyStateTracker.keyupEmitter.addListener( domEvent => {
      if ( KeyboardUtils.isKeyEvent( domEvent, KeyboardUtils.KEY_CTRL ) ) {
        this.cancel();
      }
    } );
  }

  /**
   * Sets the DerivedProperty controlling whether speech will be made with speech synthesis. The enabledProperty
   * is pushed onto provided Properties in this function, do not include it.
   * @public
   *
   * @param {Property[]} properties - list of Properties for the canSpeakProperty DerivedProperty
   * @param {function} derivation - derivation for the canSpeakProperty from properties array
   */
  setCanSpeakProperty( properties, derivation ) {
    this._canSpeakProperty.unlink( this.boundHandleCanSpeakChange );
    this._canSpeakProperty.dispose();

    properties.push( this.enabledProperty );
    this._canSpeakProperty = new DerivedProperty( properties, ( ...args ) => {
      return derivation( ...args ) && this.enabledProperty.value;
    } );

    this._canSpeakProperty.link( this.boundHandleCanSpeakChange );
  }

  /**
   * When we can no longer speak, cancel all speech to silence everything.
   * @private
   *
   * @param {boolean} canSpeak
   */
  handleCanSpeakChange( canSpeak ) {
    if ( !canSpeak ) { this.cancel(); }
  }

  /**
   * Get the available voices for the synth, and set to default.
   * @private
   */
  populateVoices() {

    // for now, only include the english voice
    this.voices = _.filter( this.synth.getVoices(), voice => {
      return voice.lang === 'en-US';
    } );

    this.voiceProperty.set( this.voices[ 0 ] );
  }

  /**
   * Implements announce so the webSpeaker can be a source of output for utteranceQueue.
   * @public
   *
   * @param {Utterance} utterance
   */
  announce( utterance ) {
    let withCancel = true;
    if ( utterance instanceof VoicingUtterance ) {
      if ( this.previousUtterance && this.previousUtterance === utterance ) {
        withCancel = utterance.cancelSelf;
      }
      else {
        withCancel = utterance.cancelOther;
      }
    }

    webSpeaker.speak( utterance, withCancel );
  }

  /**
   * Use speech synthesis to speak an utterance. No-op unless webSpeaker is initialized.
   * @public
   *
   * @param {Utterance} utterance
   * @param {boolean} withCancel - if true, any utterances remaining in the queue will be removed and this utterance
   *                               will take priority. Hopefully this works on all platforms, if it does not we
   *                               need to implement our own queing system.
   */
  speak( utterance, withCancel = true ) {
    if ( this.initialized && this._canSpeakProperty.value && !this.onHold ) {

      // only cancel the previous alert if there is something new to speak
      if ( withCancel && utterance.alert ) {
        this.synth.cancel();
      }

      // since the "end" event doesn't come through all the time after cancel() on
      // safari, we broadcast this right away to indicate that any previous speaking
      // is done
      if ( this.speakingProperty.get() && this.previousUtterance && withCancel ) {
        this.endSpeakingEmitter.emit( this.previousUtterance );
        this.speakingProperty.value = false;
      }

      // embeddding marks (for i18n) impact the output, strip before speaking
      const speechSynthUtterance = new SpeechSynthesisUtterance( stripEmbeddingMarks( utterance.getTextToAlert() ) );
      speechSynthUtterance.voice = this.voiceProperty.value;
      speechSynthUtterance.pitch = this.voicePitchProperty.value;
      speechSynthUtterance.rate = this.voiceRateProperty.value;

      // keep a reference to teh WebSpeechUtterance or Safari, so the browser
      // doesn't dispose of it before firing, see #215
      this.utterances.push( speechSynthUtterance );

      // Keep this out of the start listener so that it can be synchrounous to the UtteranceQueue draining/announcing, see bug in https://github.com/phetsims/sun/issues/699#issuecomment-831529485
      this.previousUtterance = utterance;

      const startListener = () => {
        this.startSpeakingEmitter.emit( utterance );
        this.speakingProperty.set( true );
        speechSynthUtterance.removeEventListener( 'start', startListener );
      };

      const endListener = () => {
        this.endSpeakingEmitter.emit( utterance );
        this.speakingProperty.set( false );
        speechSynthUtterance.removeEventListener( 'end', endListener );
      };

      speechSynthUtterance.addEventListener( 'start', startListener );
      speechSynthUtterance.addEventListener( 'end', endListener );

      // on safari, giving a bit of a delay to the speak request makes the `end`
      // SpeechSynthesisUtterance event come through much more consistently
      if ( platform.safari ) {
        stepTimer.setTimeout( () => {
          this.synth.speak( speechSynthUtterance );
        }, 500 );
      }
      else {
        this.synth.speak( speechSynthUtterance );
      }
    }
  }

  /**
   * Speak something initially and synchronously after some user interaction. This is helpful for a couple of cases:
   *   1) Browsers require that speech happen in response to some user interaction, with absolutely no delay.
   *   announce() is used with the utteranceQueue, which does not speak things instantly. Use this when speech is
   *   enabled, then use speak for all other usages.
   *   2) There are rare cases where we need to speak even when when canSpeakProperty is false (like speaking
   *   that voicing has been successfully turned off).
   * @public
   *
   * @param {string} utterThis
   */
  speakImmediately( utterThis ) {
    if ( this.initialized ) {

      // embidding marks (for i18n) impact the output, strip before speaking
      const utterance = new SpeechSynthesisUtterance( stripEmbeddingMarks( utterThis ) );
      utterance.voice = this.voiceProperty.value;
      utterance.pitch = this.voicePitchProperty.value;
      utterance.rate = this.voiceRateProperty.value;

      // Keep this synchrounous as it pertains to the UtteranceQueue draining/announcing, see bug in https://github.com/phetsims/sun/issues/699#issuecomment-831529485
      this.previousUtterance = utterance;

      this.synth.speak( utterance );
    }
  }

  /**
   * Stops all current speech as well and removes all utterances in the queue.
   * @public
   */
  cancel() {
    if ( this.initialized ) {
      this.synth.cancel();
    }
  }
}

const webSpeaker = new WebSpeaker();

scenery.register( 'webSpeaker', webSpeaker );
export default webSpeaker;