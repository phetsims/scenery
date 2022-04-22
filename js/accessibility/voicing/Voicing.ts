// Copyright 2021-2022, University of Colorado Boulder

/**
 * A trait for Node that supports the Voicing feature, under accessibility. Allows you to define responses for the Node
 * and make requests to speak that content using HTML5 SpeechSynthesis and the UtteranceQueue. Voicing content is
 * organized into four categories which are responsible for describing different things. Responses are stored on the
 * composed type: "ResponsePacket." See that file for details about what responses it stores. Output of this content
 * can be controlled by the responseCollector. Responses are defined as the following. . .
 *
 * - "Name" response: The name of the object that uses Voicing. Similar to the "Accessible Name" in web accessibility.
 * - "Object" response: The state information about the object that uses Voicing.
 * - "Context" response: The contextual changes that result from interaction with the Node that uses Voicing.
 * - "Hint" response: A supporting hint that guides the user toward a desired interaction with this Node.
 *
 * See ResponsePacket, as well as the property and setter documentation for each of these responses for more
 * information.
 *
 * Once this content is set, you can make a request to speak it using an UtteranceQueue with one of the provided
 * functions in this Trait. It is up to you to call one of these functions when you wish for speech to be made. The only
 * exception is on the 'focus' event. Every Node that composes Voicing will speak its responses by when it
 * receives focus.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

// Disable for the whole file
/* eslint-disable no-protected-jsdoc */

import inheritance from '../../../../phet-core/js/inheritance.js';
import ResponsePacket, { ResolvedResponse, ResponsePacketOptions, VoicingResponse } from '../../../../utterance-queue/js/ResponsePacket.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import Utterance, { IAlertable, UtteranceOptions } from '../../../../utterance-queue/js/Utterance.js';
import { Instance, InteractiveHighlighting, InteractiveHighlightingOptions, Node, scenery, SceneryListenerFunction, voicingUtteranceQueue } from '../../imports.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import TinyProperty from '../../../../axon/js/TinyProperty.js';

// Helps enforce that the utterance is defined.
function assertUtterance( utterance: Utterance | null ): asserts utterance is Utterance {
  if ( !( utterance instanceof Utterance ) ) {
    throw new Error( 'utterance is not an Utterance' );
  }
}

// An implementation class for Voicing.ts, only used in this class so that we know if we own the Utterance and can
// therefore dispose it.
class VoicingUtterance extends Utterance {
  constructor( providedOptions?: UtteranceOptions ) {
    super( providedOptions );
  }
}

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingUtterance',
  'voicingResponsePatternCollection',
  'voicingIgnoreVoicingManagerProperties',
  'voicingFocusListener'
];

type SelfOptions = {

  // see ResponsePacket.nameResponse
  voicingNameResponse?: VoicingResponse;

  // see ResponsePacket.objectResponse
  voicingObjectResponse?: VoicingResponse;

  // see ResponsePacket.contextResponse
  voicingContextResponse?: VoicingResponse;

  // see ResponsePacket.hintResponse
  voicingHintResponse?: VoicingResponse;

  // see ResponsePacket.responsePatternCollection
  voicingResponsePatternCollection?: ResponsePatternCollection;

  // see ResponsePacket.ignoreProperties
  voicingIgnoreVoicingManagerProperties?: boolean;

  // Called when this Node is focused to speak voicing responses on focus. See Voicing.defaultFocusListener for default
  // listener.
  voicingFocusListener?: SceneryListenerFunction<FocusEvent> | null;

  // The utterance to use if you want this response to be more controlled in the UtteranceQueue. This Utterance will be
  // used by all responses spoken by this class. Null to not use an Utterance.
  voicingUtterance?: Utterance | null;
};

export type VoicingOptions = SelfOptions & InteractiveHighlightingOptions;

export type SpeakingOptions = {
  utterance?: SelfOptions['voicingUtterance'];
} & {

  // In speaking options, we don't allow a ResponseCreator function, but just a string|null. The `undefined` is to
  // match on the properties because they are optional (marked with `?`)
  [PropertyName in keyof ResponsePacketOptions]: ResponsePacketOptions[PropertyName] extends ( VoicingResponse | undefined ) ?
                                                 ResolvedResponse :
                                                 ResponsePacketOptions[PropertyName];
}

/**
 * @param Type
 * @param optionsArgPosition - zero-indexed number that the options argument is provided at
 */
const Voicing = <SuperType extends Constructor>( Type: SuperType, optionsArgPosition: number ) => {

  assert && assert( _.includes( inheritance( Type ), Node ), 'Only Node subtypes should compose Voicing' );

  // Unfortunately, nothing can be private or protected in this class, see https://github.com/phetsims/scenery/issues/1340#issuecomment-1020692592
  const VoicingClass = class extends InteractiveHighlighting( Type, optionsArgPosition ) {

    // ResponsePacket that holds all the supported responses to be Voiced
    _voicingResponsePacket!: ResponsePacket;

    // The utterance that all responses are spoken through.
    _voicingUtterance: Utterance | null;

    // Called when this node is focused.
    _voicingFocusListener!: SceneryListenerFunction<FocusEvent> | null;

    // Indicates whether this Node can speak. A Node can speak if self and all of its ancestors are visible and
    // voicingVisible.
    _voicingCanSpeakProperty!: TinyProperty<boolean>;

    // A counter that keeps track of visible and voicingVisible Instances of this Node.
    // As long as this value is greater than zero, this Node can speak. See onInstanceVisibilityChange
    // and onInstanceVoicingVisibilityChange for more implementation details.
    _voicingCanSpeakCount!: number;

    // Called when `canVoiceEmitter` emits for an Instance.
    _boundInstanceCanVoiceChangeListener: ( canSpeak: boolean ) => void;

    // Called when instances of this Node change.
    _boundInstancesChangedListener!: ( instance: Instance, added: boolean ) => void;

    // Input listener that speaks content on focus. This is the only input listener added
    // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
    // response, and interaction hint.
    _speakContentOnFocusListener!: { focus: SceneryListenerFunction<FocusEvent> };

    constructor( ...args: IntentionalAny[] ) {

      const providedOptions = ( args[ optionsArgPosition ] || {} ) as VoicingOptions;

      const voicingOptions = _.pick( providedOptions, VOICING_OPTION_KEYS );
      args[ optionsArgPosition ] = _.omit( providedOptions, VOICING_OPTION_KEYS );

      super( ...args );

      // Bind the listeners on construction to be added to observables on initialize and removed on clean/dispose.
      // Instances are updated asynchronously in updateDisplay. The bind creates a new function and we need the
      // reference to persist through the completion of initialize and disposal.
      this._boundInstanceCanVoiceChangeListener = this._onInstanceCanVoiceChange.bind( this );

      this._voicingUtterance = null;

      // We only want to call this method, not any subtype implementation
      VoicingClass.prototype.initialize.call( this );

      ( this as unknown as Node ).mutate( voicingOptions );
    }

    // Separate from the constructor to support cases where Voicing is used in Poolable Nodes.
    // ...args: IntentionalAny[] because things like RichTextLink need to provide arguments to initialize, and TS complains
    // otherwise
    initialize( ...args: IntentionalAny[] ): this {

      // @ts-ignore
      super.initialize && super.initialize( args );

      // Indicates whether this Node can speak. A Node can speak if self and all of its ancestors are visible and
      // voicingVisible.
      this._voicingCanSpeakProperty = new TinyProperty<boolean>( true );

      this._voicingResponsePacket = new ResponsePacket();
      this._voicingFocusListener = this.defaultFocusListener;

      // Sets the default voicingUtterance and makes this.canSpeakProperty a dependency on its ability to announce.
      this.setVoicingUtterance( new VoicingUtterance() );

      // A counter that keeps track of visible and voicingVisible Instances of this Node. As long as this value is
      // greater than zero, this Node can speak. See onInstanceVisibilityChange and onInstanceVoicingVisibilityChange
      // for more details.
      this._voicingCanSpeakCount = 0;

      // Whenever an Instance of this Node is added or removed, add/remove listeners that will update the
      // canSpeakProperty.
      this._boundInstancesChangedListener = this._addOrRemoveInstanceListeners.bind( this );
      ( this as unknown as Node ).changedInstanceEmitter.addListener( this._boundInstancesChangedListener );

      this._speakContentOnFocusListener = {
        focus: event => {
          this._voicingFocusListener && this._voicingFocusListener( event );
        }
      };
      ( this as unknown as Node ).addInputListener( this._speakContentOnFocusListener );

      return this;
    }

    /**
     * Speak all responses assigned to this Node. Options allow you to override a responses for this particular
     * speech request. Each response is only spoken if the associated Property of responseCollector is true. If
     * all are Properties are false, nothing will be spoken.
     */
    voicingSpeakFullResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to _collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>()( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'nameResponse' ) ) {
        options.nameResponse = this._voicingResponsePacket.nameResponse;
      }
      if ( !options.hasOwnProperty( 'objectResponse' ) ) {
        options.objectResponse = this._voicingResponsePacket.objectResponse;
      }
      if ( !options.hasOwnProperty( 'contextResponse' ) ) {
        options.contextResponse = this._voicingResponsePacket.contextResponse;
      }
      if ( !options.hasOwnProperty( 'hintResponse' ) ) {
        options.hintResponse = this._voicingResponsePacket.hintResponse;
      }

      this._collectAndSpeakResponse( options );
    }

    /**
     * Speak ONLY the provided responses that you pass in with options. This will NOT speak the name, object,
     * context, or hint responses assigned to this node by default. But it allows for clarity at usages so it is
     * clear that you are only requesting certain responses. If you want to speak all of the responses assigned
     * to this Node, use voicingSpeakFullResponse().
     *
     * Each response will only be spoken if the Properties of responseCollector are true. If all of those are false,
     * nothing will be spoken.
     */
    voicingSpeakResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to _collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>()( {
        nameResponse: null,
        objectResponse: null,
        contextResponse: null,
        hintResponse: null
      }, providedOptions );

      this._collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the name response. But accepts all other responses through options. Respects responseCollector
     * Properties, so the name response may not be spoken if responseCollector.nameResponseEnabledProperty is false.
     */
    voicingSpeakNameResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to _collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>()( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'nameResponse' ) ) {
        options.nameResponse = this._voicingResponsePacket.nameResponse;
      }

      this._collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the object response. But accepts all other responses through options. Respects responseCollector
     * Properties, so the name response may not be spoken if responseCollector.objectResponseEnabledProperty is false.
     */
    voicingSpeakObjectResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to _collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>()( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'objectResponse' ) ) {
        options.objectResponse = this._voicingResponsePacket.objectResponse;
      }

      this._collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the context response. But accepts all other responses through options. Respects
     * responseCollector Properties, so the name response may not be spoken if
     * responseCollector.contextResponseEnabledProperty is false.
     */
    voicingSpeakContextResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to _collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>()( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'contextResponse' ) ) {
        options.contextResponse = this._voicingResponsePacket.contextResponse;
      }

      this._collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the hint response. But accepts all other responses through options. Respects
     * responseCollector Properties, so the hint response may not be spoken if
     * responseCollector.hintResponseEnabledProperty is false.
     */
    voicingSpeakHintResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to _collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>()( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'hintResponse' ) ) {
        options.hintResponse = this._voicingResponsePacket.hintResponse;
      }

      this._collectAndSpeakResponse( options );
    }

    /**
     * Collect responses with the responseCollector and speak the output with an UtteranceQueue.
     */
    _collectAndSpeakResponse( providedOptions?: SpeakingOptions ): void {
      this.speakContent( this.collectResponse( providedOptions ) );
    }

    /**
     * Combine all types of response into a single alertable, potentially depending on the current state of
     * responseCollector Properties (filtering what kind of responses to present in the resolved response).
     *
     * @protected // eslint-disable-line bad-typescript-text
     */
    collectResponse( providedOptions?: SpeakingOptions ): IAlertable {
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>()( {
        ignoreProperties: this._voicingResponsePacket.ignoreProperties,
        responsePatternCollection: this._voicingResponsePacket.responsePatternCollection,
        utterance: this.voicingUtterance
      }, providedOptions );

      let response: IAlertable = responseCollector.collectResponses( options ); // eslint-disable-line no-undef

      if ( options.utterance ) {
        options.utterance.alert = response;
        response = options.utterance;
      }
      return response;
    }

    /**
     * Use the provided function to create content to speak in response to input. The content is then added to the
     * back of the voicing UtteranceQueue.
     * @protected // eslint-disable-line bad-typescript-text
     */
    speakContent( content: IAlertable ): void { // eslint-disable-line no-undef

      // don't send to utteranceQueue if response is empty
      if ( content ) {
        voicingUtteranceQueue.addToBack( content );
      }
    }

    /**
     * Sets the voicingNameResponse for this Node. This is usually the label of the element and is spoken
     * when the object receives input. When requesting speech, this will only be spoken if
     * responseCollector.nameResponsesEnabledProperty is set to true.
     */
    setVoicingNameResponse( response: VoicingResponse ): void {
      this._voicingResponsePacket.nameResponse = response;
    }

    set voicingNameResponse( response: VoicingResponse ) { this.setVoicingNameResponse( response ); }

    /**
     * Get the voicingNameResponse for this Node.
     */
    getVoicingNameResponse(): ResolvedResponse {
      return this._voicingResponsePacket.nameResponse;
    }

    get voicingNameResponse(): ResolvedResponse { return this.getVoicingNameResponse(); }

    /**
     * Set the object response for this Node. This is usually the state information associated with this Node, such
     * as its current input value. When requesting speech, this will only be heard when
     * responseCollector.objectResponsesEnabledProperty is set to true.
     */
    setVoicingObjectResponse( response: VoicingResponse ) {
      this._voicingResponsePacket.objectResponse = response;
    }

    set voicingObjectResponse( response: VoicingResponse ) { this.setVoicingObjectResponse( response ); }

    /**
     * Gets the object response for this Node.
     */
    getVoicingObjectResponse(): ResolvedResponse {
      return this._voicingResponsePacket.objectResponse;
    }

    get voicingObjectResponse(): ResolvedResponse { return this.getVoicingObjectResponse(); }

    /**
     * Set the context response for this Node. This is usually the content that describes what has happened in
     * the surrounding application in response to interaction with this Node. When requesting speech, this will
     * only be heard if responseCollector.contextResponsesEnabledProperty is set to true.
     */
    setVoicingContextResponse( response: VoicingResponse ) {
      this._voicingResponsePacket.contextResponse = response;
    }

    set voicingContextResponse( response: VoicingResponse ) { this.setVoicingContextResponse( response ); }

    /**
     * Gets the context response for this Node.
     */
    getVoicingContextResponse(): ResolvedResponse {
      return this._voicingResponsePacket.contextResponse;
    }

    get voicingContextResponse(): ResolvedResponse { return this.getVoicingContextResponse(); }

    /**
     * Sets the hint response for this Node. This is usually a response that describes how to interact with this Node.
     * When requesting speech, this will only be spoken when responseCollector.hintResponsesEnabledProperty is set to
     * true.
     */
    setVoicingHintResponse( response: VoicingResponse ) {
      this._voicingResponsePacket.hintResponse = response;
    }

    set voicingHintResponse( response: VoicingResponse ) { this.setVoicingHintResponse( response ); }

    /**
     * Gets the hint response for this Node.
     */
    getVoicingHintResponse(): ResolvedResponse {
      return this._voicingResponsePacket.hintResponse;
    }

    get voicingHintResponse(): ResolvedResponse { return this.getVoicingHintResponse(); }

    /**
     * Set whether or not all responses for this Node will ignore the Properties of responseCollector. If false,
     * all responses will be spoken regardless of responseCollector Properties, which are generally set in user
     * preferences.
     */
    setVoicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ) {
      this._voicingResponsePacket.ignoreProperties = ignoreProperties;
    }

    set voicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ) { this.setVoicingIgnoreVoicingManagerProperties( ignoreProperties ); }

    /**
     * Get whether or not responses are ignoring responseCollector Properties.
     */
    getVoicingIgnoreVoicingManagerProperties(): boolean {
      return this._voicingResponsePacket.ignoreProperties;
    }

    get voicingIgnoreVoicingManagerProperties(): boolean { return this.getVoicingIgnoreVoicingManagerProperties(); }

    /**
     * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
     * additional content for each combination of response. See ResponsePatternCollection.js if you wish to use
     * a collection of string patterns that are not the default.
     */
    setVoicingResponsePatternCollection( patterns: ResponsePatternCollection ) {

      this._voicingResponsePacket.responsePatternCollection = patterns;
    }

    set voicingResponsePatternCollection( patterns: ResponsePatternCollection ) { this.setVoicingResponsePatternCollection( patterns ); }

    /**
     * Get the ResponsePatternCollection object that this Voicing Node is using to collect responses.
     */
    getVoicingResponsePatternCollection(): ResponsePatternCollection {
      return this._voicingResponsePacket.responsePatternCollection;
    }

    get voicingResponsePatternCollection(): ResponsePatternCollection { return this.getVoicingResponsePatternCollection(); }

    /**
     * Sets the utterance through which voicing associated with this Node will be spoken. By default on initialize,
     * one will be created, but a custom one can optionally be provided.
     */
    public setVoicingUtterance( utterance: Utterance ) {
      if ( this._voicingUtterance !== utterance ) {

        // `this` is not recognized as a VoicingNode, but it is because this trait can only be used with Node subtypes
        const thisVoicingNode = this as unknown as VoicingNode;

        if ( this._voicingUtterance ) {
          this._cleanVoicingUtterance();
        }

        Voicing.registerUtteranceToVoicingNode( utterance, thisVoicingNode );
        this._voicingUtterance = utterance;
      }
    }

    set voicingUtterance( utterance: Utterance ) { this.setVoicingUtterance( utterance ); }

    /**
     * Gets the utterance through which voicing associated with this Node will be spoken.
     */
    getVoicingUtterance(): Utterance {
      assertUtterance( this._voicingUtterance );
      return this._voicingUtterance;
    }

    get voicingUtterance(): Utterance { return this.getVoicingUtterance(); }

    /**
     * Get the Property indicating that this Voicing Node can speak. True when this Voicing Node and all of its
     * ancestors are visible and voicingVisible.
     */
    getVoicingCanSpeakProperty() {
      return this._voicingCanSpeakProperty;
    }

    get voicingCanSpeakProperty() { return this.getVoicingCanSpeakProperty(); }

    /**
     * Called whenever this Node is focused.
     */
    setVoicingFocusListener( focusListener: SceneryListenerFunction<FocusEvent> | null ) {
      this._voicingFocusListener = focusListener;
    }

    set voicingFocusListener( focusListener: SceneryListenerFunction<FocusEvent> | null ) { this.setVoicingFocusListener( focusListener ); }

    /**
     * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
     */
    getVoicingFocusListener(): SceneryListenerFunction<FocusEvent> | null {
      return this._voicingFocusListener;
    }

    get voicingFocusListener(): SceneryListenerFunction<FocusEvent> | null { return this.getVoicingFocusListener(); }


    /**
     * The default focus listener attached to this Node during initialization.
     */
    defaultFocusListener(): void {
      this.voicingSpeakFullResponse( {
        contextResponse: null
      } );
    }

    /**
     * Whether or not a Node composes Voicing.
     */
    get isVoicing(): boolean {
      return true;
    }

    /**
     * Detaches references that ensure this components of this Trait are eligible for garbage collection.
     */
    override dispose() {
      const thisVoicingNode = ( this as unknown as VoicingNode );
      thisVoicingNode.removeInputListener( this._speakContentOnFocusListener );
      thisVoicingNode.changedInstanceEmitter.removeListener( this._boundInstancesChangedListener );

      if ( this._voicingUtterance ) {
        this._cleanVoicingUtterance();
        this._voicingUtterance = null;
      }

      super.dispose();
    }

    clean() {
      const thisVoicingNode = ( this as unknown as VoicingNode );
      thisVoicingNode.removeInputListener( this._speakContentOnFocusListener );
      thisVoicingNode.changedInstanceEmitter.removeListener( this._boundInstancesChangedListener );

      if ( this._voicingUtterance ) {
        this._cleanVoicingUtterance();
        this._voicingUtterance = null;
      }

      // @ts-ignore
      super.clean && super.clean();
    }

    /***********************************************************************************************************/
    // PRIVATE METHODS
    /***********************************************************************************************************/

    /**
     * When visibility and voicingVisibility change such that the Instance can now speak, update the counting
     * variable that tracks how many Instances of this VoicingNode can speak. To speak the Instance must be globally\
     * visible and voicingVisible.
     */
    _onInstanceCanVoiceChange( canSpeak: boolean ): void {

      if ( canSpeak ) {
        this._voicingCanSpeakCount++;
      }
      else {
        this._voicingCanSpeakCount--;
      }

      assert && assert( this._voicingCanSpeakCount >= 0, 'the voicingCanSpeakCount should not go below zero' );
      assert && assert( this._voicingCanSpeakCount <= ( this as unknown as Node ).instances.length,
        'The voicingCanSpeakCount cannot be greater than the number of Instances.' );

      this._voicingCanSpeakProperty.value = this._voicingCanSpeakCount > 0;
    }

    /**
     * Update the canSpeakProperty and counting variable in response to an Instance of this Node being added or
     * removed.
     */
    _handleInstancesChanged( instance: Instance, added: boolean ) {
      const isVisible = instance.visible && instance.voicingVisible;
      if ( isVisible ) {

        // If the added Instance was visible and voicingVisible it should increment the counter. If the removed
        // instance is NOT visible/voicingVisible it would not have contributed to the counter so we should not
        // decrement in that case.
        this._voicingCanSpeakCount = added ? this._voicingCanSpeakCount + 1 : this._voicingCanSpeakCount - 1;
      }

      this._voicingCanSpeakProperty.value = this._voicingCanSpeakCount > 0;
    }

    /**
     * Add or remove listeners on an Instance watching for changes to visible or voicingVisible that will modify
     * the voicingCanSpeakCount. See documentation for voicingCanSpeakCount for details about how this controls the
     * voicingCanSpeakProperty.
     */
    _addOrRemoveInstanceListeners( instance: Instance, added: boolean ) {
      assert && assert( instance.voicingVisibleEmitter, 'Instance must be initialized.' );
      assert && assert( instance.visibleEmitter, 'Instance must be initialized.' );

      if ( added ) {
        // @ts-ignore - Emitters in Instance need typing
        instance.canVoiceEmitter!.addListener( this._boundInstanceCanVoiceChangeListener );
      }
      else {
        // @ts-ignore - Emitters in Instance need typing
        instance.canVoiceEmitter!.removeListener( this._boundInstanceCanVoiceChangeListener );
      }

      // eagerly update the canSpeakProperty and counting variables in addition to adding change listeners
      this._handleInstancesChanged( instance, added );
    }

    /**
     * Clean this._voicingUtterance, disposing if we own it or unregistering it if we do not.
     */
    _cleanVoicingUtterance() {
      assert && assert( this._voicingUtterance, 'A voicingUtterance must be available to clean.' );
      if ( this._voicingUtterance instanceof VoicingUtterance ) {
        this._voicingUtterance.dispose();
      }
      else {
        Voicing.unregisterUtteranceToVoicingNode( this._voicingUtterance!, this as unknown as VoicingNode );
      }
    }
  };

  /**
   * {Array.<string>} - String keys for all of the allowed options that will be set by Node.mutate( options ), in
   * the order they will be evaluated.
   * @protected // eslint-disable-line bad-typescript-text
   *
   * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
   *       cases that may apply.
   */
  VoicingClass.prototype._mutatorKeys = VOICING_OPTION_KEYS.concat( VoicingClass.prototype._mutatorKeys );
  assert && assert( VoicingClass.prototype._mutatorKeys.length === _.uniq( VoicingClass.prototype._mutatorKeys ).length, 'duplicate mutator keys in Voicing' );

  return VoicingClass;
};

Voicing.VOICING_OPTION_KEYS = VOICING_OPTION_KEYS;

/**
 * Alert an Utterance to the voicingUtteranceQueue. The Utterance must have canAnnounceProperties and hopefully
 * at least one of the Properties is a VoicingNode's canAnnounceProperty so that this Utterance is only announced
 * when the VoicingNode is globally visible and voicingVisible.
 * @static
 */
Voicing.alertUtterance = ( utterance: Utterance ) => {
  assert && assert( utterance.canAnnounceProperties.length > 0, 'canAnnounceProperties required, this Utterance might not be connected to Node in the scene graph.' );
  voicingUtteranceQueue.addToBack( utterance );
};

/**
 * Assign the voicingNode's voicingCanSpeakProperty to the Utterance so that the Utterance can only be announced
 * if the voicingNode is globally visible and voicingVisible in the display.
 * @static
 */
Voicing.registerUtteranceToVoicingNode = ( utterance: Utterance, voicingNode: VoicingNode ) => {
  const existingCanAnnounceProperties = utterance.canAnnounceProperties;
  if ( !existingCanAnnounceProperties.includes( voicingNode.voicingCanSpeakProperty ) ) {
    utterance.canAnnounceProperties = existingCanAnnounceProperties.concat( [ voicingNode.voicingCanSpeakProperty ] );
  }
};

/**
 * Remove a voicingNode's voicingCanSpeakProperty from the Utterance.
 * @static
 */
Voicing.unregisterUtteranceToVoicingNode = ( utterance: Utterance, voicingNode: VoicingNode ) => {
  const existingCanAnnounceProperties = utterance.canAnnounceProperties;
  const index = existingCanAnnounceProperties.indexOf( voicingNode.voicingCanSpeakProperty );
  assert && assert( index > -1, 'voicingNode.voicingCanSpeakProperty is not on the Utterance, was it not registered?' );
  utterance.canAnnounceProperties = existingCanAnnounceProperties.splice( index, 1 );
};

/**
 * Assign the Node's voicingVisibleProperty and visibleProperty to the Utterance so that the Utterance can only be
 * announced if the Node is visible and voicingVisible. This is LOCAL visibility and does not care about ancestors.
 * This should rarely be used, in general you should be registering an Utterance to a VoicingNode and its
 * voicingCanSpeakProperty.
 * @static
 */
Voicing.registerUtteranceToNode = ( utterance: Utterance, node: Node ) => {
  const existingCanAnnounceProperties = utterance.canAnnounceProperties;
  if ( !existingCanAnnounceProperties.includes( node.visibleProperty ) ) {
    utterance.canAnnounceProperties = utterance.canAnnounceProperties.concat( [ node.visibleProperty ] );
  }
  if ( !existingCanAnnounceProperties.includes( node.voicingVisibleProperty ) ) {
    utterance.canAnnounceProperties = utterance.canAnnounceProperties.concat( [ node.voicingVisibleProperty ] );
  }
};

/**
 * Remove a Node's voicingVisibleProperty and visibleProperty from the canAnnounceProperties of the Utterance.
 * @static
 */
Voicing.unregisterUtteranceToNode = ( utterance: Utterance, node: Node ) => {
  const existingCanAnnounceProperties = utterance.canAnnounceProperties;
  assert && assert( existingCanAnnounceProperties.includes( node.visibleProperty ) && existingCanAnnounceProperties.includes( node.voicingVisibleProperty ),
    'visibleProperty and voicingVisibleProperty were not on the Utterance, was it not registered to the node?' );

  const visiblePropertyIndex = existingCanAnnounceProperties.indexOf( node.visibleProperty );
  const withoutVisibleProperty = existingCanAnnounceProperties.splice( visiblePropertyIndex, 1 );

  const voicingVisiblePropertyIndex = withoutVisibleProperty.indexOf( node.voicingVisibleProperty );
  const withoutBothProperties = existingCanAnnounceProperties.splice( voicingVisiblePropertyIndex, 1 );

  utterance.canAnnounceProperties = withoutBothProperties;
};

// Export a type that lets you check if your Node is composed with Voicing.
const wrapper = () => Voicing( Node, 0 );
export type VoicingNode = InstanceType<ReturnType<typeof wrapper>>;

scenery.register( 'Voicing', Voicing );
export default Voicing;
